import os
import io
import json
import struct
import random
import gc
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Any, List, Optional
from torch.utils.data import Dataset
import torch.distributed as dist
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import warnings
from collections import OrderedDict
from PIL import Image, UnidentifiedImageError
from omnivae_generation.trainer.data import (
    _build_image_transform,
    _normalize_prompt_max_sequence_length,
    add_tokenized_prompt_fields,
    collate_tokenized_prompt_fields,
    maybe_format_chat_prompt,
    maybe_tokenize_prompt_to_tensors,
)

try:
    import orjson
except ImportError:
    class _OrjsonCompat:
        JSONDecodeError = json.JSONDecodeError

        @staticmethod
        def loads(payload):
            if isinstance(payload, (bytes, bytearray, memoryview)):
                payload = bytes(payload).decode("utf-8")
            return json.loads(payload)

    orjson = _OrjsonCompat()

Image.MAX_IMAGE_PIXELS = None

# v3 索引只保存 file_idx + 主 jsonl 的字节范围，行号通过每个文件的样本前缀和反推。
INDEX_CACHE_VERSION = "byte_offsets_v2_i4"
METADATA_DT = np.dtype([
    ("file_idx", "i4"),
    ("byte_start", "i4"),
    ("byte_length", "i4"),
])


def _count_lines_task(task):
    full_path, file_idx = task
    try:
        count = 0
        last_byte = None
        with open(full_path, 'rb') as f:
            buf_size = 1024 * 1024
            read_f = f.read
            buf = read_f(buf_size)
            while buf:
                count += buf.count(b'\n')
                last_byte = buf[-1]
                buf = read_f(buf_size)
        if last_byte is not None and last_byte != ord("\n"):
            count += 1
        return file_idx, count
    except Exception:
        return file_idx, 0


def _build_cache_key(root: str) -> str:
    resolved_root = str(Path(root).expanduser().resolve())
    return hashlib.sha1(resolved_root.encode("utf-8")).hexdigest()[:12]

def _coerce_prompt_candidates(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []

    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item]

    return []


def choose_relaion_prompt(sample: dict, recaption_prob: float = 0.0) -> str:
    caption_candidates = _coerce_prompt_candidates(sample.get("caption"))
    if not caption_candidates:
        caption_candidates = _coerce_prompt_candidates(sample.get("text"))

    recaption_candidates = _coerce_prompt_candidates(sample.get("recaption"))

    if caption_candidates and recaption_candidates:
        if random.random() < recaption_prob:
            return random.choice(recaption_candidates)
        return random.choice(caption_candidates)

    if recaption_candidates:
        return random.choice(recaption_candidates)

    if caption_candidates:
        return random.choice(caption_candidates)

    return ""

class RelaionDataset(Dataset):
    def __init__(
        self,
        root: str,  # 对应 master_path
        slave_path: str,
        base_image_path: str,
        split: str = "train",
        image_size: int = 256,
        center_crop: bool = False,
        random_flip: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: str = ".cache/relaion",
        repeat: int = 1,
        recaption_prob: float = 0.0,
        tokenizer=None,
        include_raw_pixel_values: bool = False,
        prompt_max_sequence_length: int | None = None,
        **kwargs # 吸收多余参数
    ):
        super().__init__()
        if not 0.0 <= recaption_prob <= 1.0:
            raise ValueError(f"recaption_prob must be within [0, 1], got {recaption_prob!r}")

        self.master_path = root
        self.slave_path = slave_path
        self.base_image_path = base_image_path
        self.repeat = repeat
        self.recaption_prob =recaption_prob
        self.tokenizer = tokenizer
        self.include_raw_pixel_values = bool(include_raw_pixel_values)
        self.empty_prompt = maybe_format_chat_prompt("", tokenizer)
        self.prompt_max_sequence_length = _normalize_prompt_max_sequence_length(prompt_max_sequence_length)
        self.empty_prompt_tokens = maybe_tokenize_prompt_to_tensors(
            self.empty_prompt,
            tokenizer,
            self.prompt_max_sequence_length,
        )
        
        # 统一 transform 逻辑
        self.transform = _build_image_transform(
            image_size,
            center_crop,
            random_flip,
            include_raw_pixel_values=self.include_raw_pixel_values,
        )
        
        # 使用版本化 + root hash 的缓存名，避免和旧 line_num 索引冲突。
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _build_cache_key(root)
        self.file_list_path = os.path.join(
            cache_dir,
            f"relaion_file_list_{INDEX_CACHE_VERSION}_{cache_key}_{split}.json",
        )
        self.mmap_path = os.path.join(
            cache_dir,
            f"relaion_metadata_{INDEX_CACHE_VERSION}_{cache_key}_{split}.npy",
        )
        
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        self._prepare_index()
        
        # 加载索引
        with open(self.file_list_path, 'r') as f:
            file_list_payload = json.load(f)
        self.file_names = file_list_payload["files"]
        self.file_line_counts = np.asarray(file_list_payload["line_counts"], dtype=np.int64)
        self.file_sample_offsets = np.zeros(len(self.file_line_counts) + 1, dtype=np.int64)
        self.file_sample_offsets[1:] = np.cumsum(self.file_line_counts, dtype=np.int64)
        indexed_num_samples = os.path.getsize(self.mmap_path) // METADATA_DT.itemsize
        if int(self.file_sample_offsets[-1]) != indexed_num_samples:
            raise ValueError(
                f"Relaion index metadata mismatch: line_counts sum to {int(self.file_sample_offsets[-1])}, "
                f"but metadata file contains {indexed_num_samples} samples."
            )

        self.num_samples = indexed_num_samples
        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)
            
        self.metadata = np.memmap(self.mmap_path, dtype=METADATA_DT, mode='r', shape=(self.num_samples,))
        self._jsonl_offset_cache = OrderedDict()   # path -> offsets(list[int]), only used for slave_path fallback.
        self._max_cached_jsonl_files = 2           # 不要太大，防止 worker 内存膨胀
        self._max_retries = 8
        self._max_getitems_concurrency = 4

    def _prepare_index(self):
        index_ready = os.path.exists(self.file_list_path) and os.path.exists(self.mmap_path)
        if self.is_main_process and not index_ready:
            print(f"[Relaion] Indexing {self.master_path}...")
            all_rel_paths = []
            subdirs = sorted([d for d in os.listdir(self.master_path) if os.path.isdir(os.path.join(self.master_path, d))])
            for sd in subdirs:
                m_subdir = os.path.join(self.master_path, sd)
                files = sorted([f for f in os.listdir(m_subdir) if f.endswith(".jsonl")])
                all_rel_paths.extend([os.path.join(sd, f) for f in files])

            full_paths = [os.path.join(self.master_path, p) for p in all_rel_paths]
            tasks = [(full_path, i) for i, full_path in enumerate(full_paths)]
            counts = [0] * len(all_rel_paths)
            with ProcessPoolExecutor(max_workers=32) as executor:
                for f_idx, count in tqdm(executor.map(_count_lines_task, tasks), total=len(tasks), desc="Scanning"):
                    counts[f_idx] = count

            with open(self.file_list_path, 'w') as f:
                json.dump({"files": all_rel_paths, "line_counts": counts}, f)

            total_lines = sum(counts)
            mmap_array = np.memmap(self.mmap_path, dtype=METADATA_DT, mode='w+', shape=(total_lines,))
            curr = 0
            for f_idx, count in tqdm(
                enumerate(counts),
                total=len(counts),
                desc="Building byte-offset index",
            ):
                if count > 0:
                    file_size = os.path.getsize(full_paths[f_idx])
                    if file_size > np.iinfo(np.int32).max:
                        raise ValueError(
                            f"JSONL file exceeds int32 offset range: {full_paths[f_idx]} ({file_size} bytes)."
                        )

                    byte_starts = np.empty(count, dtype="i4")
                    byte_lengths = np.empty(count, dtype="i4")
                    start = 0
                    actual_count = 0
                    with open(full_paths[f_idx], "rb") as f:
                        while actual_count < count:
                            line = f.readline()
                            if not line:
                                break
                            end = f.tell()
                            byte_starts[actual_count] = start
                            byte_lengths[actual_count] = end - start
                            start = end
                            actual_count += 1

                    if actual_count != count:
                        raise RuntimeError(
                            f"Index mismatch for {full_paths[f_idx]}: counted {count} lines, rebuilt {actual_count}."
                        )

                    mmap_array[curr : curr + count]["file_idx"] = f_idx
                    mmap_array[curr : curr + count]["byte_start"] = byte_starts
                    mmap_array[curr : curr + count]["byte_length"] = byte_lengths
                    curr += count
            mmap_array.flush()
            print(f"[Relaion] Total samples found: {total_lines}")

        if dist.is_initialized():
            dist.barrier()

    def __len__(self):
        return self.num_samples * self.repeat
    
    def _get_line_offsets(self, path: str):
        offsets = self._jsonl_offset_cache.get(path)
        if offsets is not None:
            self._jsonl_offset_cache.move_to_end(path)
            return offsets

        # 构建当前文件的行起始偏移
        offsets = [0]
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                offsets.append(f.tell())

        self._jsonl_offset_cache[path] = offsets
        self._jsonl_offset_cache.move_to_end(path)

        # LRU 淘汰，防止缓存无限增长
        while len(self._jsonl_offset_cache) > self._max_cached_jsonl_files:
            self._jsonl_offset_cache.popitem(last=False)

        return offsets

    def _read_line_bytes(self, path: str, line_num: int) -> bytes:
        offsets = self._get_line_offsets(path)
        if line_num < 0 or line_num >= len(offsets) - 1:
            raise IndexError(f"line_num out of range: {line_num} for {path}")

        start = offsets[line_num]
        end = offsets[line_num + 1]
        return self._read_byte_range(path, start, end)

    def _read_byte_range(self, path: str, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range [{start}, {end}) for {path}")
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def _read_package_bytes(self, path: str, offset: int, expected_length: int) -> bytes:
        with open(path, "rb", buffering=expected_length + 1024) as fh:
            fh.seek(offset + 32)  # skip hash
            ext_len = struct.unpack("B", fh.read(1))[0]
            fh.read(ext_len)      # skip ext
            data_len = struct.unpack("I", fh.read(4))[0]
            if expected_length != data_len:
                raise ValueError(
                    f"error in binary data, data length in bin read out {data_len=} different from meta info {expected_length=}"
                )
            return fh.read(data_len)

    def resolve_sample_location(self, sample_index: int):
        meta = self.metadata[sample_index]
        file_idx = int(meta["file_idx"])
        byte_start = int(meta["byte_start"])
        byte_length = int(meta["byte_length"])
        # Metadata is laid out file-by-file, so the global sample index minus the file's
        # starting sample offset recovers the original line number for slave_path fallback.
        return (
            self.file_names[file_idx],
            int(sample_index - self.file_sample_offsets[file_idx]),
            byte_start,
            byte_start + byte_length,
        )

    def _load_sample_payload(self, actual_idx: int):
        rel_path, line_num, byte_start, byte_end = self.resolve_sample_location(actual_idx)

        master_jsonl = os.path.join(self.master_path, rel_path)
        m_line = self._read_byte_range(master_jsonl, byte_start, byte_end)
        dm = orjson.loads(m_line)

        prompt = choose_relaion_prompt(dm, recaption_prob=self.recaption_prob)

        if not prompt and self.slave_path:
            slave_jsonl = os.path.join(self.slave_path, rel_path)
            s_line = self._read_line_bytes(slave_jsonl, line_num)
            ds = orjson.loads(s_line)
            prompt = choose_relaion_prompt(ds, recaption_prob=self.recaption_prob)

        package_name = dm["package"]
        offset = int(dm["offset"])
        length = int(dm["length"])
        package_path = os.path.join(self.base_image_path, os.path.dirname(rel_path), package_name)

        img_bytes = self._read_package_bytes(package_path, offset, length)

        return {
            "rel_path": rel_path,
            "line_num": line_num,
            "prompt": prompt,
            "img_bytes": img_bytes,
        }

    def _finalize_sample(self, rel_path: str, line_num: int, prompt: str, img_bytes: bytes):

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Corrupt EXIF data.*",
                category=UserWarning,
            )
            img = Image.open(io.BytesIO(img_bytes))
            img.load()

        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        img = img.convert("RGB")

        transformed = self.transform(img)
        if self.include_raw_pixel_values:
            pixel_values, raw_pixel_values = transformed
        else:
            pixel_values = transformed
            raw_pixel_values = None

        prompt = maybe_format_chat_prompt(prompt, self.tokenizer)
        sample = {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "empty_prompt": self.empty_prompt,
            "image_id": f"{rel_path}_{line_num}",
        }
        add_tokenized_prompt_fields(
            sample,
            prompt=prompt,
            empty_prompt=self.empty_prompt,
            tokenizer=self.tokenizer,
            max_sequence_length=self.prompt_max_sequence_length,
            empty_prompt_tokens=self.empty_prompt_tokens,
        )
        if raw_pixel_values is not None:
            sample["raw_pixel_values"] = raw_pixel_values
        return sample

    def _getitem_with_retry(self, idx: int):
        last_error = None

        for attempt in range(self._max_retries):
            try:
                actual_idx = idx % self.num_samples if attempt == 0 else random.randint(0, self.num_samples - 1)
                payload = self._load_sample_payload(actual_idx)
                return self._finalize_sample(**payload)
            except (
                FileNotFoundError,
                KeyError,
                IndexError,
                struct.error,
                OSError,
                ValueError,
                UnidentifiedImageError,
                orjson.JSONDecodeError,
            ) as e:
                last_error = e
                print(last_error)
                continue

        raise RuntimeError(f"Failed to fetch sample after {self._max_retries} retries. Last error: {last_error}")

    def _getitems_threaded(self, indices: List[int]):
        max_workers = max(1, min(len(indices), self._max_getitems_concurrency))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._getitem_with_retry, indices))

    def __getitem__(self, idx):
        return self._getitem_with_retry(idx)

    def __getitems__(self, indices):
        if not indices:
            return []

        if len(indices) > 1 and self._max_getitems_concurrency > 1:
            samples = self._getitems_threaded(indices)
        else:
            samples = [self._getitem_with_retry(idx) for idx in indices]

        if not gc.isenabled():
            gc.collect()
        return samples
        
        
def collate_relaion_samples(batch):
    """
    针对 Relaion 数据集的打包函数
    确保返回的字典包含训练脚本中 batch["pixel_values"] 和 batch["prompts"] 所需的键
    """
    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "empty_prompts": [item["empty_prompt"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        # Relaion 没有 image_path, label_texts, synsets, 
        # 为了保持下游兼容性，可以将 image_id 填充到这些字段或留空
        "image_paths": [item.get("image_id", "") for item in batch], 
        "label_texts": ["" for _ in batch],
        "synsets": ["" for _ in batch],
    }
    if "raw_pixel_values" in batch[0]:
        collated["raw_pixel_values"] = torch.stack([item["raw_pixel_values"] for item in batch])
    collate_tokenized_prompt_fields(collated, batch)
    return collated
