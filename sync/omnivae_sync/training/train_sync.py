import math
import time
import shutil
import logging
from pathlib import Path
from glob import glob

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from omnivae_sync.utils.logger import LoggerWithTBoard
from omnivae_sync.training.train_utils import (EarlyStopper, AverageMeter,
                                 broadcast_obj, get_batch_sizes, get_curr_time_w_random_shift, get_datasets,
                                 get_device, get_loaders, get_lr_scheduler,
                                 get_model, get_optimizer, get_transforms,
                                 init_ddp, is_master, load_ckpt,
                                 make_backward_and_optim_step, prepare_inputs,
                                 set_seed, toggle_mode, verbose_epoch_progress, apply_fn_recursive,
                                 verbose_test_progress)
from omnivae_sync.utils.utils import show_cfg_diffs


def _load_clean2_bad_clip_ids() -> set[str]:
    """
    “第二种 clean”过滤对齐 `dataset.vggsound.VGGSoundSparsePickedCleanTest.filter_bad_examples`：
      bad = union(
        ./data/filtered_examples_vggsound/*.txt,
        ./data/filtered_examples_vggsound_extra/*.txt
      )
    返回 clip_id（不带 .mp4 后缀）。
    """
    bad: set[str] = set()
    for base in ("./data/filtered_examples_vggsound", "./data/filtered_examples_vggsound_extra"):
        base_path = Path(base)
        if not base_path.exists():
            continue
        for p in sorted(glob(str(base_path / "*.txt"))):
            try:
                for line in open(p, "r", encoding="utf-8").read().splitlines():
                    s = line.strip()
                    if s:
                        bad.add(Path(s).stem)
            except Exception:
                continue
    return bad


def _apply_mask_recursive(x, keep_mask: torch.Tensor):
    """
    对 tensor/dict/list 递归应用 batch 维度 mask。
    只用于评测阶段的日志聚合；若结构不支持则原样返回。
    """
    try:
        if isinstance(x, torch.Tensor):
            return x[keep_mask]
        if isinstance(x, dict):
            return {k: _apply_mask_recursive(v, keep_mask) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            out = [_apply_mask_recursive(v, keep_mask) for v in x]
            return type(x)(out)
    except Exception:
        return x
    return x


def train(cfg):
    init_ddp(cfg)
    global_rank = cfg.training.global_rank
    world_size = cfg.training.world_size

    if cfg.action == 'ft_avsync_model_for_syncability':
        target_key = 'sync_target'
    else:
        target_key = 'offset_target'

    # makes iterations faster if your inputs are of a fixed size
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    device, num_gpus = get_device(cfg)

    # ckpt_path was created only for the master (to keep it the same), now we broadcast it to each worker
    cfg.ckpt_path = broadcast_obj(cfg.ckpt_path, global_rank, device)
    assert hasattr(cfg, 'ckpt_path'), f'No ckpt_path in the config: {cfg} for worker {global_rank}'

    set_seed(cfg.training.seed)  # same seed for all workers for model init
    model, model_without_ddp = get_model(cfg, device)

    # Append estimated segment counts to run_name (pool mode only)
    _inner = model.module if hasattr(model, 'module') else model
    if hasattr(_inner, 'skip_temporal_pool') and not _inner.skip_temporal_pool:
        if getattr(_inner, '_latent_per_segment', 0) > 0:
            _v_compress = getattr(_inner.video_vae, 'temporal_compress_factor', 4)
            _subsample = getattr(_inner, 'video_subsample_step', 1)
            _tail_sec = float(getattr(_inner, 'tail_extra_sec', 0.0))
            _effective_crop = cfg.data.crop_len_sec + _tail_sec
            _causal_extra = 1
            _T_input = int(cfg.data.target_fps * _effective_crop) + _causal_extra
            _T_sub = (_T_input + _subsample - 1) // _subsample
            _T_latent = (_T_sub - 1) // _v_compress
            _total_seg = max(1, round(_T_latent / _inner._latent_per_segment))
            _tail_seg = _inner._tail_segments
            _final_seg = _total_seg - _tail_seg
            _seg_tag = f'_seg{_final_seg}'
            if _tail_seg > 0:
                _seg_tag += f't{_tail_seg}'
            cfg.run_name = cfg.run_name + _seg_tag
            if is_master(global_rank):
                logging.info(f'run_name updated with segment info: {cfg.run_name} '
                             f'(est_latent={_T_latent}, total_seg={_total_seg}, '
                             f'tail_seg={_tail_seg}, final_seg={_final_seg})')

    logger = LoggerWithTBoard(global_rank, cfg)

    if is_master(global_rank):
        logging.info(f'Config: \n{OmegaConf.to_yaml(cfg)}')

    optimizer = get_optimizer(cfg, model, num_gpus)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    set_seed(cfg.training.seed + global_rank)
    batch_sizes = get_batch_sizes(cfg, num_gpus)
    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms)
    loaders = get_loaders(cfg, datasets, batch_sizes)

    logger.log_param_num(global_rank, model)

    early_stopper = EarlyStopper(cfg.training.patience, cfg.training.to_max_metric, cfg.training.metric_name)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.training.use_half_precision)

    # this chunk has a complicate logic but it simply loads pre-trained ckpt during finetuning/resuming/test
    if cfg.training.run_test_only or cfg.training.resume or cfg.training.finetune:
        ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        start_epoch = ckpt['epoch']
        ckpt_cfg = ckpt['args']
        ckpt_metrics = ckpt['metrics']
        if cfg.training.finetune:
            # if ft, allow for the current model to be a bit different from the one in the ckpt, eg heads
            _ckpt_load_status = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
            if len(_ckpt_load_status.missing_keys) > 0 or len(_ckpt_load_status.unexpected_keys) > 0:
                logging.warning(f'ckpt load status: {_ckpt_load_status}')
                logging.warning('Check if the above missing keys are expected. If so, ignore this warning.'
                                'Otherwise (e.g. lots of unmatched keys), fix the ckpt or the model.'
                                'Expectable keys could be heads during fine-tuning')
            # restarting training counters if the ckpt is used to init weights rather than continuing training
            logging.info(f'Finetuning from: {ckpt_cfg.ckpt_path} on {datasets["train"].__class__.__name__}')
            ckpt_metrics[cfg.training.metric_name] = 0 if cfg.training.to_max_metric else float('inf')
            start_epoch = 0
            # saving the diff between the current cfg and the one in the ckpt
            show_cfg_diffs(ckpt_cfg, cfg, Path(cfg.ckpt_path).parent / 'cfg_diffs.diff')
        else:
            model_without_ddp.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            if 'lr_scheduler' in ckpt:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if cfg.training.resume:
                start_epoch += 1
        early_stopper.set_best_metrics(ckpt_metrics)
    else:
        start_epoch = 0

    # don't do training loops if a user wants to only probe the model on the test set
    num_epochs = 0 if cfg.training.run_test_only else cfg.training.num_epochs

    if cfg.training.compile:
        model = torch.compile(model)

    loss_fn = cfg.training.get('loss_fn', None)

    # loop over the train and validation multiple times (typical PT boilerplate)
    for epoch in range(start_epoch, num_epochs):

        # keep it in this order, otherwise the resume will be done with `epoch + 1`
        phases_to_run_on = ['train', 'valid']

        for phase in phases_to_run_on:
            # does model.eval() or .train() on appropriate submodules
            toggle_mode(cfg, model, phase)

            # init runnining results
            running_results = dict()
            losses_m = {'loss_total': AverageMeter()}
            batch_time_m = AverageMeter()
            data_time_m = AverageMeter()
            iter_time_m = AverageMeter()
            end = time.time()

            if dist.is_initialized():
                loaders[phase].sampler.set_epoch(epoch)

            # how many times to iterate through a evaluation se (makes estimates more robust for small dsets)
            if phase == 'valid' and 'VGGSoundSparsePicked' in loaders[phase].dataset.__class__.__name__:
                iter_times = cfg.data.get('iter_times', 1)
            else:
                iter_times = 1

            for it in range(iter_times):

                # resetting batch / data time meters per log window
                data_time_m.reset()
                batch_time_m.reset()
                iter_time_m.reset()

                # making a list of equally-spaced indices to update running results, as dense logging fails
                # on large datasets with OOM
                total_samples = len(loaders[phase].dataset)
                log_max_items = cfg.logging.get('log_max_items', total_samples)
                # equally-spaced ids `set` removes duplicates (eg 0, 0, 1, 1) if log_max_items > total_samples
                ids_to_cache = set(torch.arange(0, total_samples, total_samples/log_max_items).long().tolist())
                # although the above ignores that all ids are spread across workers, it should still work fine
                # as the ids are <= worker_samples = total_samples / world_size

                num_samples = 0
                for i, batch in enumerate(loaders[phase]):
                    # unfortunately, I had to use this to avoid GPU mem error on the second iteration
                    if i == 1:
                        torch.cuda.empty_cache()
                    iter_step = epoch * len(loaders[phase]) + i
                    # zero the parameter gradients; same as optimizer.zero_grad() but more mem efficient
                    model.zero_grad(set_to_none=True)

                    # sends targets and inputs to cuda
                    aud, vid, targets = prepare_inputs(batch, device, phase)

                    num_samples += len(vid) * cfg.training.world_size

                    # saves recontructed input to the model during the first iteration (detects bugs)
                    # doing it before torch.no_grad because some reconstruction transforms require it
                    _skip_viz = cfg.action in ('train_avsync_model_vae',)
                    if iter_step == 0 and phase in ['train', 'valid'] and not _skip_viz:
                        if is_master(global_rank):
                            logger.vizualize_input(vid, aud, batch, iter_step, phase, cfg)
                        # just wait for the master to finish
                        if dist.is_initialized():
                            dist.barrier()

                    data_time_m.update(time.time() - end)

                    # gradient and half-precision toggles
                    with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                        with torch.set_grad_enabled(phase == 'train'):
                            loss, logits = model(vid, aud, targets[target_key], loss_fn=loss_fn)

                    _loss_is_bad = not torch.isfinite(loss).all()
                    if _loss_is_bad:
                        logging.warning(f'Worker (#{global_rank}): Loss is not finite ({loss.item():.4g}), '
                                        f'zeroing loss for batch {i} at epoch {epoch}')
                        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)) * 0.0

                    grad_norm = None
                    if phase == 'train':
                        grad_norm = make_backward_and_optim_step(cfg, loss, model, optimizer, scaler, lr_scheduler)

                    if _loss_is_bad:
                        end = time.time()
                        continue

                    batch_time_m.update(time.time() - end)

                    # gathering results in one place to iterate on this later
                    iter_results = dict(
                        logits=[apply_fn_recursive(logits, lambda x: x.detach().cpu())],
                        targets=[apply_fn_recursive(targets[target_key], lambda x: x.cpu())],
                        path_suffix=[p.split("/")[-1] for p in batch['path']],
                    )
                    losses_m['loss_total'].update(loss.item(), len(vid))
                    if 'offset_sec' in targets:
                        iter_results['offset_sec'] = [apply_fn_recursive(targets['offset_sec'], lambda x: x.cpu())]

                    if is_master(global_rank) and i % cfg.logging.log_frequency == 0:
                        # iter logging (making it a bit more sparse for faster tboard loading)
                        iter_loss = losses_m['loss_total'].val
                        logger.log_iter_loss(iter_loss, iter_step, phase, prefix='total')
                        if phase == 'train':
                            logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], iter_step)
                            if grad_norm is not None:
                                logger.add_scalar('train/grad_norm', grad_norm, iter_step)
                            # --- VAE feature diagnostics ---
                            _inner = model.module if hasattr(model, 'module') else model
                            diag = getattr(_inner, '_diag_stats', {})
                            for k, v in diag.items():
                                logger.add_scalar(f'diag/{k}', v, iter_step)
                        if cfg.logging.get('vis_segment_sim', False) and phase in ['train', 'valid']:
                            # visualize segments (but 10 times less often coarsely)
                            if i % (cfg.logging.log_frequency * 10) == 0:
                                toggle_mode(cfg, model, 'valid')
                                logger.vizualize_segment_sim(cfg, model, vid, aud, iter_step, phase)
                                if phase == 'train':
                                    toggle_mode(cfg, model, 'train')
                        batch_size = len(vid)
                        samples_per_epoch = len(loaders[phase].dataset)
                        percent_done = 100.0 * (i+1) / len(loaders[phase])
                        sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
                        samples_per_s = batch_size * world_size / batch_time_m.avg
                        samples_per_s_per_gpu = batch_size / batch_time_m.avg
                        msg = f'{phase} ({epoch})'
                        msg += f' [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_done:.0f}%)]'
                        msg += f' Data (t): {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                        msg += f' Batch (t): {batch_time_m.val:.3f} ({batch_time_m.avg:.3f},'
                        msg += f' {samples_per_s:#g}/s, {samples_per_s_per_gpu:#g}/s/gpu)'
                        msg += f' Log (t): ({iter_time_m.avg:.3f})'
                        msg += f' Loss: {iter_loss:.3f}'
                        logging.info(msg)

                    # update running results; doing it uniformly-coarsely to avoid OOM issues (log_max_items)
                    if i in ids_to_cache:
                        for res_k, res_v in iter_results.items():
                            default_obj = [] if isinstance(res_v, list) else 0
                            running_results[res_k] = running_results.get(res_k, default_obj) + iter_results[res_k]

                    iter_time_m.update(time.time() - end)
                    end = time.time()

                if is_master(global_rank):
                    logging.info(f'({phase}) Done {it} iterations out of {iter_times}')

            # logs epoch metrics to tensorboard/wandb
            for loss_name, loss_meter in losses_m.items():
                running_results[loss_name] = loss_meter.avg
            save_csv_with_preds = cfg.action == 'train_avsync_multitask_oos_off'
            # NOTE: the dataset is 'train'
            epoch_loss, metrics = verbose_epoch_progress(global_rank, logger, running_results, phase, epoch,
                                                         datasets['train'], save_csv_with_preds)

            # Early stopping check and maybe update
            if phase == cfg.training.early_stop_phase:
                has_model_improved = early_stopper.is_new_model_better_than_curr(metrics)
                if has_model_improved:
                    early_stopper.reset_patience(global_rank, metrics)
                else:
                    early_stopper.increment_patience(global_rank)

            # logging ckpts: latest after each training epoch, best after early-stop-phase if improved
            if is_master(global_rank):
                if phase == 'train':  # no need to save again after the valid loop
                    # NOTE: model is saved with the best `cfg.training.early_stop_phase` metrics, not train's
                    logger.log_latest_model(model_without_ddp, scaler, epoch_loss,
                                            epoch, optimizer, lr_scheduler, early_stopper.best_metrics, cfg)
                # Early stopping update
                if phase == cfg.training.early_stop_phase and has_model_improved:
                    logger.log_best_model(model_without_ddp, scaler, epoch_loss, epoch, optimizer,
                                          lr_scheduler, metrics, cfg)

            # wait for other workers to get here
            if dist.is_initialized():
                dist.barrier()

        if early_stopper.triggered:
            if is_master(global_rank):
                logging.info(f'Training is early stopped @ {epoch}; RANK: {global_rank}')
            break

    if is_master(global_rank):
        logging.info('Finished Training')
    if dist.is_initialized():
        dist.barrier()

    # don't do testing if a user wants to only train the model
    if cfg.get('skip_test', False):
        # finish wandb logging
        if is_master(global_rank):
            logging.info('Skipping testing')
            logger.finish_wandb_logging()
        return

    # Testing the model
    phase = 'test'
    cfg.training.finetune = False
    # loading the best model
    ckpt_epoch, best_metrics_val = load_ckpt(cfg, model_without_ddp, optimizer, scaler, lr_scheduler)
    best_metric_val = best_metrics_val[cfg.training.metric_name]
    if is_master(global_rank):
        logging.info(f'Loading the best model from {cfg.ckpt_path}')
        logging.info(f'Best metric: {best_metric_val}')
        logging.info((f'The model was trained for {ckpt_epoch} epochs.'))
        logging.info(f'Copying the checkpoint to *_eXX.pt from {cfg.ckpt_path}')
        p = Path(cfg.ckpt_path)
        curr_t = get_curr_time_w_random_shift()
        shutil.copyfile(str(p), str((p.parent / f'{p.stem}_e{ckpt_epoch}_t{curr_t}').with_suffix(p.suffix)))
    if dist.is_initialized():
        dist.barrier()

    model.eval()

    # init runnining results
    # running_results = dict(logits=[], targets=[], loss_total=0)
    running_results = dict()
    # 可选：一次跑 full，同时统计 clean2 子集（避免跑两遍）
    compute_clean2_subset = bool(cfg.get("eval", {}).get("compute_clean2_subset", False))
    bad_clean2 = _load_clean2_bad_clip_ids() if compute_clean2_subset else set()
    running_results_clean2 = dict() if compute_clean2_subset else None
    losses_m = {'loss_total': AverageMeter()}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    if dist.is_initialized():
        loaders[phase].sampler.set_epoch(ckpt_epoch)

    # how many times to iterate through a evaluation dataset (makes estimates more robust for small datasets)
    iter_times = cfg.data.get('iter_times', 1)
    for it in range(iter_times):

        # resetting batch / data time meters per log window
        batch_time_m.reset()
        data_time_m.reset()

        num_samples = 0
        for i, batch in enumerate(loaders[phase]):
            # sends inputs and targets to cuda
            aud, vid, targets = prepare_inputs(batch, device, phase)
            # zero the parameter gradients
            data_time_m.update(time.time() - end)
            optimizer.zero_grad()
            # gradient and half-precision toggles
            with torch.set_grad_enabled(False):
                with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                    loss, logits = model(vid, aud, targets[target_key], loss_fn=loss_fn)

            num_samples += len(vid) * cfg.training.world_size
            batch_time_m.update(time.time() - end)

            # it is easier to keep logits and targets in the same format
            if cfg.action == 'ft_avsync_model_for_attribution':
                logits = {'video': logits[0], 'audio': logits[1]}
            elif cfg.action == 'ft_avsync_model_for_attribution_n_synchability':
                logits = {'offset': logits[0], 'video': logits[1][0], 'audio': logits[1][1]}
            elif cfg.action == 'train_avsync_multitask_oos_off':
                logits = {'oos': logits[0], 'offset': logits[1]}

            # gathering results in one place to iterate on this later
            iter_results = dict(
                logits=[apply_fn_recursive(logits, lambda x: x.detach().cpu())],
                targets=[apply_fn_recursive(targets[target_key], lambda x: x.cpu())],
            )
            losses_m['loss_total'].update(loss.item(), len(vid))

            for res_k, res_v in iter_results.items():
                default_obj = [] if isinstance(res_v, list) else 0
                running_results[res_k] = running_results.get(res_k, default_obj) + iter_results[res_k]

            # 同步累积 clean2 子集
            if compute_clean2_subset:
                assert running_results_clean2 is not None
                try:
                    paths = batch.get('path', [])
                    keep = []
                    for p in paths:
                        clip_id = Path(str(p)).stem
                        keep.append(clip_id not in bad_clean2)
                    if any(keep):
                        keep_mask = torch.tensor(keep, dtype=torch.bool)
                        iter_results_c = {
                            "logits": [_apply_mask_recursive(iter_results["logits"][0], keep_mask)],
                            "targets": [_apply_mask_recursive(iter_results["targets"][0], keep_mask)],
                        }
                        for res_k, res_v in iter_results_c.items():
                            default_obj = [] if isinstance(res_v, list) else 0
                            running_results_clean2[res_k] = running_results_clean2.get(res_k, default_obj) + res_v
                except Exception:
                    # 不影响 full 主流程
                    pass

            if is_master(global_rank) and i % cfg.logging.log_frequency == 0:
                batch_size = len(vid)
                samples_per_epoch = len(loaders[phase].dataset)
                ratio_done = (i+1) / len(loaders[phase])
                sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))
                msg = f'{phase} ({ckpt_epoch}) [{num_samples:>{sample_digits}}/{samples_per_epoch} ({ratio_done:.0%})]'
                if data_time_m is not None:
                    msg += f' Data (t): {data_time_m.avg:.3f}'
                if batch_time_m is not None:
                    samples_per_s = batch_size * world_size / batch_time_m.val
                    samples_per_s_per_gpu = batch_size / batch_time_m.val
                    msg += f' Batch (t): {batch_time_m.avg:.3f}, {samples_per_s:.1f}/s, {samples_per_s_per_gpu:.1f}/s/gpu'
                logging.info(msg)

            end = time.time()

        if is_master(global_rank):
            logging.info(f'Done iterations {it+1} / {iter_times}')

        if dist.is_initialized():
            dist.barrier()

        if is_master(global_rank):
            logging.info(f'Passed Barrier {it+1} / {iter_times}')

    # logs test metrics to tensorboard/wandb
    for loss_name, loss_meter in losses_m.items():
        running_results[loss_name] = loss_meter.avg
    verbose_test_progress(global_rank, logger, cfg, running_results, ckpt_epoch)

    # 额外：clean2 子集 metrics（同一次 run）
    if compute_clean2_subset:
        # NOTE: 不要在函数内 import `is_master`，否则会把它变成局部变量，导致上面早期使用时报
        # UnboundLocalError。这里直接复用模块顶部 import 的 `is_master`。
        from omnivae_sync.training.train_utils import calc_cls_metrics, gather_dict  # local import
        running_results_clean2 = gather_dict(running_results_clean2 or {})
        if is_master(global_rank) and running_results_clean2 and len(running_results_clean2.get("logits", [])) > 0:
            try:
                logits_c = torch.cat(running_results_clean2["logits"]).float()
                targets_c = torch.cat(running_results_clean2["targets"]).long()
                metrics_c = calc_cls_metrics(
                    targets_c, logits_c, add_doubt_cls=cfg.training.get("add_doubt_cls", False), prefix="clean2_"
                )
                metrics_c = {k: round(v, 4) for k, v in metrics_c.items()}
                logging.info(f'Clean2 metrics on {phase} set: {metrics_c}')
                # also log to tboard/wandb with prefix
                logger.log_test_metrics(metrics_c, dict(cfg), ckpt_epoch, prefix="clean2")
            except Exception as e:
                logging.warning(f'Failed to compute/log clean2 metrics: {type(e).__name__}: {e}')

    # finish wandb logging
    if is_master(global_rank):
        logger.finish_wandb_logging()
