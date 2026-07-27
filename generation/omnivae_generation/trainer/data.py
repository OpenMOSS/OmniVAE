from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DEFAULT_PROMPT_TEMPLATES = ["a high-quality photo of {label}"]


def maybe_format_chat_prompt(prompt: str, tokenizer) -> str:
    if tokenizer is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _normalize_prompt_max_sequence_length(value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"prompt_max_sequence_length must be positive, got {value!r}.")
    return parsed


def maybe_tokenize_prompt_to_tensors(prompt: str, tokenizer, max_sequence_length: int | None):
    max_sequence_length = _normalize_prompt_max_sequence_length(max_sequence_length)
    if tokenizer is None or max_sequence_length is None:
        return None
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded.input_ids[0].to(dtype=torch.long).cpu(),
        "attention_mask": encoded.attention_mask[0].to(dtype=torch.bool).cpu(),
    }


def add_tokenized_prompt_fields(
    sample: dict,
    *,
    prompt: str,
    empty_prompt: str,
    tokenizer,
    max_sequence_length: int | None,
    empty_prompt_tokens=None,
) -> None:
    prompt_tokens = maybe_tokenize_prompt_to_tensors(prompt, tokenizer, max_sequence_length)
    if prompt_tokens is None:
        return
    if empty_prompt_tokens is None:
        empty_prompt_tokens = maybe_tokenize_prompt_to_tensors(empty_prompt, tokenizer, max_sequence_length)
    assert empty_prompt_tokens is not None
    sample["prompt_input_ids"] = prompt_tokens["input_ids"]
    sample["prompt_attention_mask"] = prompt_tokens["attention_mask"]
    sample["empty_prompt_input_ids"] = empty_prompt_tokens["input_ids"]
    sample["empty_prompt_attention_mask"] = empty_prompt_tokens["attention_mask"]


def collate_tokenized_prompt_fields(collated: dict, batch: list[dict]) -> None:
    if not batch or "prompt_input_ids" not in batch[0]:
        return
    for field_name in (
        "prompt_input_ids",
        "prompt_attention_mask",
        "empty_prompt_input_ids",
        "empty_prompt_attention_mask",
    ):
        collated[field_name] = torch.stack([item[field_name] for item in batch])


def _normalize_caption_key(key: str) -> str:
    normalized = str(key).strip().replace("\\", "/")
    normalized = normalized.removeprefix("./")
    for prefix in ("train/", "val/", "test/", "CLS-LOC/train/", "CLS-LOC/val/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    for suffix in (".jpeg", ".jpg", ".png", ".webp", ".JPEG", ".JPG", ".PNG", ".WEBP"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _coerce_caption_value(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def load_caption_map(path: Optional[str]) -> Dict[str, str | List[str]]:
    if not path:
        return {}

    caption_path = Path(path)
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file does not exist: {caption_path}")

    suffix = caption_path.suffix.lower()
    caption_map: Dict[str, str | List[str]] = {}

    def add_item(key: str, value) -> None:
        normalized_key = _normalize_caption_key(key)
        normalized_value = _coerce_caption_value(value)
        if normalized_key and normalized_value:
            caption_map[normalized_key] = normalized_value

    if suffix == ".json":
        with caption_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key, value in payload.items():
                add_item(key, value)
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                key = item.get("image_id") or item.get("file_name") or item.get("path") or item.get("id")
                value = item.get("caption") or item.get("prompt") or item.get("text")
                if key is not None:
                    add_item(key, value)
    elif suffix == ".jsonl":
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("image_id") or item.get("file_name") or item.get("path") or item.get("id")
                value = item.get("caption") or item.get("prompt") or item.get("text")
                if key is not None:
                    add_item(key, value)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with caption_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for item in reader:
                key = item.get("image_id") or item.get("file_name") or item.get("path") or item.get("id")
                value = item.get("caption") or item.get("prompt") or item.get("text")
                if key is not None:
                    add_item(key, value)
    else:
        raise ValueError(f"Unsupported caption file format: {caption_path}")

    return caption_map


class PairedImageTransform:
    def __init__(self, image_size: int, center_crop: bool, random_flip: bool, include_raw_pixel_values: bool = False):
        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.include_raw_pixel_values = bool(include_raw_pixel_values)
        self.interpolation = InterpolationMode.BILINEAR

    def __call__(self, image: Image.Image):
        image = TF.resize(image, self.image_size, interpolation=self.interpolation, antialias=True)
        if self.center_crop:
            image = TF.center_crop(image, [self.image_size, self.image_size])
        else:
            top, left, height, width = transforms.RandomCrop.get_params(image, [self.image_size, self.image_size])
            image = TF.crop(image, top, left, height, width)
        if self.random_flip and random.random() < 0.5:
            image = TF.hflip(image)

        raw_pixel_values = TF.to_tensor(image)
        pixel_values = TF.normalize(raw_pixel_values, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        if not self.include_raw_pixel_values:
            return pixel_values
        return pixel_values, raw_pixel_values


def _build_image_transform(image_size: int, center_crop: bool, random_flip: bool, include_raw_pixel_values: bool = False):
    return PairedImageTransform(
        image_size=image_size,
        center_crop=center_crop,
        random_flip=random_flip,
        include_raw_pixel_values=include_raw_pixel_values,
    )


def _load_synset_mapping(dataset_root: Path, use_full_label: bool) -> Dict[int, Dict[str, str]]:
    mapping_path = dataset_root / "LOC_synset_mapping.txt"
    idx_to_entry: Dict[int, Dict[str, str]] = {}
    with mapping_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            synset, label_text = line.split(" ", 1)
            label_text = label_text.strip()
            primary_label = label_text if use_full_label else label_text.split(",")[0].strip()
            idx_to_entry[index] = {
                "synset": synset,
                "label_text": primary_label,
                "full_label_text": label_text,
            }
    return idx_to_entry


def _load_val_synsets(dataset_root: Path) -> Dict[str, str]:
    solution_path = dataset_root / "LOC_val_solution.csv"
    image_to_synset: Dict[str, str] = {}
    with solution_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["ImageId"].strip()
            prediction = row["PredictionString"].strip()
            if not prediction:
                continue
            synset = prediction.split()[0]
            image_to_synset[image_id] = synset
    return image_to_synset


class ImageNetTextToImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        center_crop: bool = False,
        random_flip: bool = True,
        max_samples: Optional[int] = None,
        captions_file: Optional[str] = None,
        prompt_templates: Optional[Iterable[str]] = None,
        use_full_label: bool = False,
        tokenizer=None,
        include_raw_pixel_values: bool = False,
        prompt_max_sequence_length: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(root).expanduser().resolve()
        self.split = split
        self.tokenizer = tokenizer
        self.empty_prompt = maybe_format_chat_prompt("", tokenizer)
        self.prompt_max_sequence_length = _normalize_prompt_max_sequence_length(prompt_max_sequence_length)
        self.empty_prompt_tokens = maybe_tokenize_prompt_to_tensors(
            self.empty_prompt,
            tokenizer,
            self.prompt_max_sequence_length,
        )
        self.prompt_templates = list(prompt_templates or DEFAULT_PROMPT_TEMPLATES)
        self.caption_map = load_caption_map(captions_file)
        self.include_raw_pixel_values = bool(include_raw_pixel_values)
        self.transform = _build_image_transform(
            image_size=image_size,
            center_crop=center_crop,
            random_flip=random_flip,
            include_raw_pixel_values=self.include_raw_pixel_values,
        )
        self.idx_to_class = _load_synset_mapping(self.dataset_root, use_full_label=use_full_label)
        self.synset_to_class = {value["synset"]: value for value in self.idx_to_class.values()}
        self.val_synsets = _load_val_synsets(self.dataset_root)
        self.entries = self._load_entries()
        if max_samples is not None:
            self.entries = self.entries[: max_samples]

    def _load_entries(self) -> List[Dict[str, str]]:
        imageset_root = self.dataset_root / "ILSVRC" / "ImageSets" / "CLS-LOC"
        data_root = self.dataset_root / "ILSVRC" / "Data" / "CLS-LOC"

        if self.split == "train":
            split_file = imageset_root / "train_cls.txt"
        elif self.split == "val":
            split_file = imageset_root / "val.txt"
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        entries: List[Dict[str, str]] = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if self.split == "train":
                    relative_id, sample_index = line.split()
                    synset = relative_id.split("/", 1)[0]
                    class_entry = self.synset_to_class[synset]
                    image_path = data_root / "train" / f"{relative_id}.JPEG"
                    image_id = Path(relative_id).name
                    class_index = sample_index
                else:
                    relative_id, sample_index = line.split()
                    synset = self.val_synsets[relative_id]
                    class_entry = self.synset_to_class[synset]
                    image_path = data_root / "val" / f"{relative_id}.JPEG"
                    image_id = relative_id
                    class_index = sample_index

                entries.append(
                    {
                        "image_path": str(image_path),
                        "image_id": image_id,
                        "relative_id": relative_id,
                        "class_index": class_index,
                        "synset": class_entry["synset"],
                        "label_text": class_entry["label_text"],
                        "full_label_text": class_entry["full_label_text"],
                    }
                )
        return entries

    def _lookup_caption(self, entry: Dict[str, str]) -> Optional[str]:
        candidate_keys = [
            entry["relative_id"],
            entry["image_id"],
            Path(entry["image_path"]).name,
            Path(entry["image_path"]).stem,
            f"{entry['synset']}/{entry['image_id']}",
        ]

        for key in candidate_keys:
            value = self.caption_map.get(_normalize_caption_key(key))
            if isinstance(value, list):
                return random.choice(value)
            if value:
                return value
        return None

    def _build_prompt(self, entry: Dict[str, str]) -> str:
        caption = self._lookup_caption(entry)
        if caption:
            return caption

        label = entry["label_text"]
        template = random.choice(self.prompt_templates)
        return template.format(label=label)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = Path(entry["image_path"])
        with Image.open(image_path) as image:
            image = exif_transpose(image).convert("RGB")
            transformed = self.transform(image)
        if self.include_raw_pixel_values:
            pixel_values, raw_pixel_values = transformed
        else:
            pixel_values = transformed
            raw_pixel_values = None
        prompt = maybe_format_chat_prompt(self._build_prompt(entry), self.tokenizer)

        sample = {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "empty_prompt": self.empty_prompt,
            "image_id": entry["image_id"],
            "image_path": str(image_path),
            "label_text": entry["label_text"],
            "synset": entry["synset"],
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


def collate_samples(batch):
    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "empty_prompts": [item["empty_prompt"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "label_texts": [item["label_text"] for item in batch],
        "synsets": [item["synset"] for item in batch],
    }
    if "raw_pixel_values" in batch[0]:
        collated["raw_pixel_values"] = torch.stack([item["raw_pixel_values"] for item in batch])
    collate_tokenized_prompt_fields(collated, batch)
    return collated
