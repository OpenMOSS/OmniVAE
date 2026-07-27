# Inference

Use single-video inference to predict the audio-video offset class for a video
crop. The script can inject a known offset before prediction, which is useful
for checking whether a checkpoint reacts as expected.

## Command

```bash
bash scripts/infer/predict_offset.sh \
  --ckpt_path /path/to/sync_model.pt \
  --cfg_path /path/to/cfg-sync.yaml \
  --vid_path /path/to/video.mp4 \
  --offset_sec 0.0 \
  --v_start_i_sec 0.0 \
  --device cuda:0 \
  --topk 5
```

`--cfg_path` is optional when the checkpoint contains a saved `args` config.
Pass it explicitly when evaluating a checkpoint produced by another training
layout or when you want to override paths.

## Python Entrypoint

```bash
python -m omnivae_sync.infer.predict_offset \
  --ckpt_path /path/to/sync_model.pt \
  --cfg_path configs/sync_24fps_nonspeech_vae.yaml \
  --vid_path /path/to/video.mp4 \
  --offset_sec 0.4 \
  av_vae_config="$VAE_CFG" \
  vae_pretrained="$VAE_CKPT"
```

Unknown trailing arguments are interpreted as OmegaConf overrides. This is the
same style used by training.

## Output

The script prints the target offset and top predicted offset classes:

```text
Target offset: 0.00s (grid=0.00s, index=10)
Top predictions:
  index= 10 offset=  0.00s prob=0.8123 logit=4.1120
```

Use `--json` for machine-readable output.
