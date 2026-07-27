#!/usr/bin/env bash

# ================= 环境初始化 =================
set -eo pipefail

echo "===================== start train ====================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
dry_run="${OMNIVAE_DRY_RUN:-0}"

export OMNIVAE_REPO_ROOT="${OMNIVAE_REPO_ROOT:-${REPO_ROOT}}"
export OMNIVAE_CKPT_ROOT="${OMNIVAE_CKPT_ROOT:-${REPO_ROOT}/ckpts}"
export OMNIVAE_DATA_ROOT="${OMNIVAE_DATA_ROOT:-${REPO_ROOT}/data}"
export OMNIVAE_EXP_ROOT="${OMNIVAE_EXP_ROOT:-${REPO_ROOT}/exp}"
export OMNIVAE_SEMANTIC_MODEL="${OMNIVAE_SEMANTIC_MODEL:-${OMNIVAE_CKPT_ROOT}/qwen3_avencoder_service}"

if [ -n "${OMNIVAE_ENV_SH:-}" ]; then
    # Optional user-local environment setup, e.g. export OMNIVAE_ENV_SH=~/env.sh.
    source "${OMNIVAE_ENV_SH}"
fi

if [ -n "${OMNIVAE_CONDA_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${OMNIVAE_CONDA_ENV}"
    else
        echo "Error: OMNIVAE_CONDA_ENV is set but conda is not available in PATH"
        exit 1
    fi
fi

cd "${REPO_ROOT}"
which python

# ================= 解析参数 =================
config=""
gpu_ids=""
debug_mode=0
debug_ip="localhost"
debug_port=32431
continue_train="--continue_train"
valid_only_flag=""
checkpoint_flag=""
loss_override_flags=""
lambda_group_video=""
lambda_group_audio=""
lambda_group_contrastive=""
lambda_segment_contrastive=""
lambda_global_contrastive=""
lambda_video_kl=""
lambda_video_lpips=""
lambda_audio_kl=""
spatial_pool_mode=""
spatial_merge_factor=""
segment_count=""
num_negatives=""
num_negative_videos=""
same_long_video_priority_flag=""
same_long_video_num_negatives=""
num_negatives_with_sibling=""
num_negatives_no_sibling=""
segment_temporal_pool_mode=""
global_temporal_pool_mode=""
transformer_nhead=""
contrastive_transformer_layers=""
spatial_transformer_layers=""
segment_transformer_layers=""
global_transformer_layers=""
contrastive_module_size=""
spatial_module_size=""
segment_module_size=""
global_module_size=""
cnn_num_blocks_per_stage=""
cnn_kernel_size=""
use_sdpa_flag=""
lr=""
batch_size=""
use_ema_flag=""
contrastive_use_mean_flag=""
val_segment_num_negatives=""
val_segment_num_negative_videos=""
val_global_num_negatives=""
val_contrastive_max_samples=""
eval_video_recon_flag=""
eval_audio_recon_flag=""
eval_contrastive_flag=""
eval_contrastive_in_all_flag=""
exp_name_suffix=""
exp_name=""
exp_name_arg=""
reset_scheduler_on_resume_flag=""
pretrained_checkpoint=""
pretrained_video_checkpoint=""
pretrained_audio_checkpoint=""
keep_audio_vae_pretrained_flag=""
global_contrastive_start_steps=""
video_distill_start_step=""
audio_distill_start_step=""
segment_avclip_start_steps=""
segment_count_weights=""
freeze_vae_encoders_flag=""
spatial_transform_mode=""
spatial_roundtrip_short_edge=""
train_metadata_path=""
grad_log_steps=""
adaptive_loss_balance=""
adaptive_balance_audio_ratio=""
adaptive_balance_contrastive_ratio=""
adaptive_loss_balance_by_uncertainty=""
uncertainty_warmup_steps=""
adaptive_loss_balance_by_gradient=""
gradient_balance_video_ratio=""
gradient_balance_audio_ratio=""
gradient_balance_clamp_max=""
gradient_balance_interval=""
dtype=""
video_vae_dtype=""
audio_vae_dtype=""
contrastive_dtype=""
num_frames=""
gradient_checkpointing=""
max_grad_norm=""
use_semantic_distill_flag=""
semantic_model_path=""
semantic_api_url=""
encoder_fps=""
encoder_resolution=""
distill_vision_layer=""
distill_audio_layer=""
distill_vision_layer=""
distill_audio_layer=""
lambda_distill_image_cosine=""
lambda_distill_image_distance=""
lambda_distill_video_cosine=""
lambda_distill_video_distance=""
lambda_distill_audio_t_axis=""
lambda_distill_audio_d_axis=""
lambda_group_distill=""
distill_margin_cosine=""
distill_margin_distance=""
distill_w_hyper=""
distill_audio_type=""
distill_proj_type=""
distill_proj_layers=""
distill_proj_hidden_dim=""
distill_use_conv3d=""
distill_proj_before_agg=""
distill_dim_schedule=""
distill_use_sampled=""
distill_spatial_norm=""
distill_spatial_norm_gamma=""
distill_use_dist_matrix=""
adaptive_distill_balance=""
adaptive_distill_use_gradient=""
adaptive_distill_video_ratio=""
adaptive_distill_audio_ratio=""
distill_upload_mode=""
distill_video_gpu_map=""
distill_image_gpu_id=""
distill_audio_gpu_id=""
distill_num_upload_workers=""
distill_processor_path=""
qk_norm_flag=""
contrastive_type=""
contrastive_embed_dim=""
contrastive_nhead=""
self_attn_layers=""
cross_attn_layers=""
max_audio_tokens_per_seg=""
max_spatial_h=""
max_spatial_w=""
contrastive_dim_feedforward=""
contrastive_dropout=""
warmup_steps=""
max_steps=""
pretrained_contrastive_checkpoint=""
pretrained_disc_checkpoint=""
pretrained_disc_load_optim_flag=""
video_loss_clamp_flag=""
video_recon_clamp_max=""
video_lpips_clamp_max=""
video_kl_clamp_max=""
video_learn_logvar_flag=""
video_logvar_init=""
gradient_accumulation_steps=""
# Adaptive loss balance v2
adaptive_loss_balance_v2=""
adaptive_anchor_source=""
adaptive_anchor_ema_decay=""
adaptive_anchor_warmup_steps=""
adaptive_scale_clamp_min=""
adaptive_scale_clamp_max=""
adaptive_ratio_video=""
adaptive_ratio_audio=""
adaptive_ratio_contrastive=""
# Stage1 overrides for adaptive v2 (active while video_vae is phase-frozen)
adaptive_anchor_source_stage1=""
adaptive_ratio_video_stage1=""
adaptive_ratio_audio_stage1=""
adaptive_ratio_contrastive_stage1=""
# Stage2 gradient-balance hybrid (switch after video VAE unfreeze)
adaptive_v2_stage2_use_gradient_flag=""
adaptive_v2_stage2_blend_steps=""
gradient_ratio_video_stage2=""
gradient_ratio_audio_stage2=""
# Phase freezing video VAE
freeze_video_vae_flag=""
freeze_video_vae_until_step=""
freeze_audio_vae_flag=""
freeze_audio_vae_until_step=""
freeze_audio_encoder_flag=""
freeze_video_encoder_flag=""
# Contrastive gradient scaling
contrastive_grad_scale_video=""
contrastive_grad_scale_audio=""
# Per-module learning rates
lr_video_vae=""
lr_audio_vae=""
lr_contrastive_head=""
lr_distill_proj=""
lr_video_logvar=""
# Dedicated warmup+cosine schedule for the video_vae param group
lr_video_vae_warmup_steps=""
lr_video_vae_total_steps=""
lr_video_vae_start_step=""
lr_video_vae_min_ratio=""
# Dedicated warmup+cosine schedule for the audio_vae param group (symmetric)
lr_audio_vae_warmup_steps=""
lr_audio_vae_total_steps=""
lr_audio_vae_start_step=""
lr_audio_vae_min_ratio=""
# Eval / save cadence
eval_steps=""
save_steps=""
# Video backbone overrides
video_model_name=""
video_model_config=""
pretrained_video_model_path=""
# Audio discriminator (LSGAN)
use_audio_disc_flag=""
audio_disc_start_step=""
lambda_audio_adv=""
lambda_audio_feature_matching=""
lr_disc=""
disc_max_grad_norm=""
disc_dtype=""
# Video discriminator (CausalVAE-style 3D PatchGAN, alternating G/D)
use_video_disc_flag=""
video_disc_start_step=""
lambda_video_adv=""
video_disc_loss_type=""
video_disc_adaptive_weight_flag=""
video_disc_adaptive_weight_max=""
video_disc_lazy_threshold=""
distill_every_steps_flag=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            gpu_ids="$2"; shift 2 ;;
        --debug)
            debug_mode=1; shift ;;
        --debug_ip)
            debug_ip="$2"; shift 2 ;;
        --debug_port)
            debug_port="$2"; shift 2 ;;
        --no_continue)
            continue_train=""; shift ;;
        --valid_only)
            valid_only_flag="--valid_only"; shift ;;
        --checkpoint)
            checkpoint_flag="--checkpoint $2"; shift 2 ;;
        --use_video_recon)
            loss_override_flags="${loss_override_flags} --use_video_recon"; shift ;;
        --no_video_recon)
            loss_override_flags="${loss_override_flags} --no-use_video_recon"; shift ;;
        --use_audio_recon)
            loss_override_flags="${loss_override_flags} --use_audio_recon"; shift ;;
        --no_audio_recon)
            loss_override_flags="${loss_override_flags} --no-use_audio_recon"; shift ;;
        --use_segment_contrastive)
            loss_override_flags="${loss_override_flags} --use_segment_contrastive"; shift ;;
        --no_segment_contrastive)
            loss_override_flags="${loss_override_flags} --no-use_segment_contrastive"; shift ;;
        --use_global_contrastive)
            loss_override_flags="${loss_override_flags} --use_global_contrastive"; shift ;;
        --no_global_contrastive)
            loss_override_flags="${loss_override_flags} --no-use_global_contrastive"; shift ;;
        --freeze_vae_encoders)
            freeze_vae_encoders_flag="--freeze_vae_encoders"; shift ;;
        --no_freeze_vae_encoders)
            freeze_vae_encoders_flag="--no-freeze_vae_encoders"; shift ;;
        --lambda_group_video)
            lambda_group_video="--lambda_group_video $2"; shift 2 ;;
        --lambda_group_audio)
            lambda_group_audio="--lambda_group_audio $2"; shift 2 ;;
        --lambda_group_contrastive)
            lambda_group_contrastive="--lambda_group_contrastive $2"; shift 2 ;;
        --lambda_segment_contrastive)
            lambda_segment_contrastive="--lambda_segment_contrastive $2"; shift 2 ;;
        --lambda_global_contrastive)
            lambda_global_contrastive="--lambda_global_contrastive $2"; shift 2 ;;
        --lambda_video_kl)
            lambda_video_kl="--lambda_video_kl $2"; shift 2 ;;
        --lambda_video_lpips)
            lambda_video_lpips="--lambda_video_lpips $2"; shift 2 ;;
        --lambda_audio_kl)
            lambda_audio_kl="--lambda_audio_kl $2"; shift 2 ;;
        --spatial_pool_mode)
            spatial_pool_mode="--spatial_pool_mode $2"; shift 2 ;;
        --spatial_merge_factor)
            spatial_merge_factor="--spatial_merge_factor $2"; shift 2 ;;
        --segment_count)
            segment_count="--segment_count $2"; shift 2 ;;
        --num_negatives)
            num_negatives="--num_negatives $2"; shift 2 ;;
        --num_negative_videos)
            num_negative_videos="--num_negative_videos $2"; shift 2 ;;
        --same_long_video_priority)
            same_long_video_priority_flag="--same_long_video_priority"; shift ;;
        --no_same_long_video_priority)
            same_long_video_priority_flag="--no-same_long_video_priority"; shift ;;
        --same_long_video_num_negatives)
            same_long_video_num_negatives="--same_long_video_num_negatives $2"; shift 2 ;;
        --num_negatives_with_sibling)
            num_negatives_with_sibling="--num_negatives_with_sibling $2"; shift 2 ;;
        --num_negatives_no_sibling)
            num_negatives_no_sibling="--num_negatives_no_sibling $2"; shift 2 ;;
        --segment_temporal_pool_mode)
            segment_temporal_pool_mode="--segment_temporal_pool_mode $2"; shift 2 ;;
        --global_temporal_pool_mode)
            global_temporal_pool_mode="--global_temporal_pool_mode $2"; shift 2 ;;
        --contrastive_transformer_layers)
            contrastive_transformer_layers="--contrastive_transformer_layers $2"; shift 2 ;;
        --transformer_nhead)
            transformer_nhead="--transformer_nhead $2"; shift 2 ;;
        --spatial_transformer_layers)
            spatial_transformer_layers="--spatial_transformer_layers $2"; shift 2 ;;
        --segment_transformer_layers)
            segment_transformer_layers="--segment_transformer_layers $2"; shift 2 ;;
        --global_transformer_layers)
            global_transformer_layers="--global_transformer_layers $2"; shift 2 ;;
        --contrastive_module_size)
            contrastive_module_size="--contrastive_module_size $2"; shift 2 ;;
        --spatial_module_size)
            spatial_module_size="--spatial_module_size $2"; shift 2 ;;
        --segment_module_size)
            segment_module_size="--segment_module_size $2"; shift 2 ;;
        --global_module_size)
            global_module_size="--global_module_size $2"; shift 2 ;;
        --cnn_num_blocks_per_stage)
            cnn_num_blocks_per_stage="--cnn_num_blocks_per_stage $2"; shift 2 ;;
        --cnn_kernel_size)
            cnn_kernel_size="--cnn_kernel_size $2"; shift 2 ;;
        --use_sdpa)
            use_sdpa_flag="--use_sdpa"; shift ;;
        --no_sdpa)
            use_sdpa_flag="--no-use_sdpa"; shift ;;
        --lr)
            lr="--lr $2"; shift 2 ;;
        --batch_size)
            batch_size="--batch_size $2"; shift 2 ;;
        --use_ema)
            use_ema_flag="--use_ema"; shift ;;
        --no_ema)
            use_ema_flag="--no-use_ema"; shift ;;
        --contrastive_use_mean)
            contrastive_use_mean_flag="--contrastive_use_mean"; shift ;;
        --contrastive_no_mean)
            contrastive_use_mean_flag="--no-contrastive_use_mean"; shift ;;
        --val_segment_num_negatives)
            val_segment_num_negatives="--val_segment_num_negatives $2"; shift 2 ;;
        --val_segment_num_negative_videos)
            val_segment_num_negative_videos="--val_segment_num_negative_videos $2"; shift 2 ;;
        --val_global_num_negatives)
            val_global_num_negatives="--val_global_num_negatives $2"; shift 2 ;;
        --val_contrastive_max_samples)
            val_contrastive_max_samples="--val_contrastive_max_samples $2"; shift 2 ;;
        --eval_video_recon)
            eval_video_recon_flag="--eval_video_recon"; shift ;;
        --no_eval_video_recon|--no-eval_video_recon)
            eval_video_recon_flag="--no-eval_video_recon"; shift ;;
        --eval_audio_recon)
            eval_audio_recon_flag="--eval_audio_recon"; shift ;;
        --no_eval_audio_recon|--no-eval_audio_recon)
            eval_audio_recon_flag="--no-eval_audio_recon"; shift ;;
        --eval_contrastive)
            eval_contrastive_flag="--eval_contrastive"; shift ;;
        --no_eval_contrastive|--no-eval_contrastive)
            eval_contrastive_flag="--no-eval_contrastive"; shift ;;
        --eval_contrastive_in_all)
            eval_contrastive_in_all_flag="--eval_contrastive_in_all"; shift ;;
        --no_eval_contrastive_in_all|--no-eval_contrastive_in_all)
            eval_contrastive_in_all_flag="--no-eval_contrastive_in_all"; shift ;;
        --exp_name_suffix)
            exp_name_suffix="--exp_name_suffix $2"; shift 2 ;;
        --exp_name)
            exp_name="$2"; exp_name_arg="--exp_name $2"; shift 2 ;;
        --reset_scheduler_on_resume)
            reset_scheduler_on_resume_flag="--reset_scheduler_on_resume"; shift ;;
        --no_reset_scheduler_on_resume|--no-reset_scheduler_on_resume)
            reset_scheduler_on_resume_flag="--no-reset_scheduler_on_resume"; shift ;;
        --pretrained_checkpoint)
            pretrained_checkpoint="--pretrained_checkpoint $2"; shift 2 ;;
        --pretrained_video_checkpoint)
            pretrained_video_checkpoint="--pretrained_video_checkpoint $2"; shift 2 ;;
        --pretrained_audio_checkpoint)
            pretrained_audio_checkpoint="--pretrained_audio_checkpoint $2"; shift 2 ;;
        --video_model_name)
            video_model_name="--video_model_name $2"; shift 2 ;;
        --video_model_config)
            video_model_config="--video_model_config $2"; shift 2 ;;
        --pretrained_video_model_path)
            pretrained_video_model_path="--pretrained_video_model_path $2"; shift 2 ;;
        --keep_audio_vae_pretrained)
            keep_audio_vae_pretrained_flag="--keep_audio_vae_pretrained"; shift ;;
        --no_keep_audio_vae_pretrained)
            keep_audio_vae_pretrained_flag="--no-keep_audio_vae_pretrained"; shift ;;
        --global_contrastive_start_steps)
            global_contrastive_start_steps="--global_contrastive_start_steps $2"; shift 2 ;;
        --video_distill_start_step)
            video_distill_start_step="--video_distill_start_step $2"; shift 2 ;;
        --audio_distill_start_step)
            audio_distill_start_step="--audio_distill_start_step $2"; shift 2 ;;
        --segment_avclip_start_steps)
            segment_avclip_start_steps="--segment_avclip_start_steps $2"; shift 2 ;;
        --segment_count_weights)
            segment_count_weights="--segment_count_weights $2"; shift 2 ;;
        --spatial_transform_mode)
            spatial_transform_mode="--spatial_transform_mode $2"; shift 2 ;;
        --spatial_roundtrip_short_edge)
            spatial_roundtrip_short_edge="--spatial_roundtrip_short_edge $2"; shift 2 ;;
        --train_metadata_path)
            train_metadata_path="--train_metadata_path $2"; shift 2 ;;
        --grad_log_steps)
            grad_log_steps="--grad_log_steps $2"; shift 2 ;;
        --adaptive_loss_balance)
            adaptive_loss_balance="--adaptive_loss_balance"; shift ;;
        --no_adaptive_loss_balance)
            adaptive_loss_balance="--no-adaptive_loss_balance"; shift ;;
        --adaptive_balance_audio_ratio)
            adaptive_balance_audio_ratio="--adaptive_balance_audio_ratio $2"; shift 2 ;;
        --adaptive_balance_contrastive_ratio)
            adaptive_balance_contrastive_ratio="--adaptive_balance_contrastive_ratio $2"; shift 2 ;;
        --adaptive_loss_balance_by_uncertainty)
            adaptive_loss_balance_by_uncertainty="--adaptive_loss_balance_by_uncertainty"; shift ;;
        --no_adaptive_loss_balance_by_uncertainty)
            adaptive_loss_balance_by_uncertainty="--no-adaptive_loss_balance_by_uncertainty"; shift ;;
        --uncertainty_warmup_steps)
            uncertainty_warmup_steps="--uncertainty_warmup_steps $2"; shift 2 ;;
        --adaptive_loss_balance_by_gradient)
            adaptive_loss_balance_by_gradient="--adaptive_loss_balance_by_gradient"; shift ;;
        --no_adaptive_loss_balance_by_gradient)
            adaptive_loss_balance_by_gradient="--no-adaptive_loss_balance_by_gradient"; shift ;;
        --gradient_balance_video_ratio)
            gradient_balance_video_ratio="--gradient_balance_video_ratio $2"; shift 2 ;;
        --gradient_balance_audio_ratio)
            gradient_balance_audio_ratio="--gradient_balance_audio_ratio $2"; shift 2 ;;
        --gradient_balance_clamp_max)
            gradient_balance_clamp_max="--gradient_balance_clamp_max $2"; shift 2 ;;
        --gradient_balance_interval)
            gradient_balance_interval="--gradient_balance_interval $2"; shift 2 ;;
        --dtype)
            dtype="--dtype $2"; shift 2 ;;
        --video_vae_dtype)
            video_vae_dtype="--video_vae_dtype $2"; shift 2 ;;
        --audio_vae_dtype)
            audio_vae_dtype="--audio_vae_dtype $2"; shift 2 ;;
        --contrastive_dtype)
            contrastive_dtype="--contrastive_dtype $2"; shift 2 ;;
        --num_frames)
            num_frames="--num_frames $2"; shift 2 ;;
        --gradient_checkpointing)
            gradient_checkpointing="--gradient_checkpointing"; shift ;;
        --no_gradient_checkpointing)
            gradient_checkpointing="--no-gradient_checkpointing"; shift ;;
        --max_grad_norm)
            max_grad_norm="--max_grad_norm $2"; shift 2 ;;
        --use_semantic_distill)
            use_semantic_distill_flag="--use_semantic_distill"; shift ;;
        --no_semantic_distill)
            use_semantic_distill_flag="--no-use_semantic_distill"; shift ;;
        --semantic_model_path)
            semantic_model_path="--semantic_model_path $2"; shift 2 ;;
        --semantic_api_url)
            semantic_api_url="--semantic_api_url $2"; shift 2 ;;
        --encoder_fps)
            encoder_fps="--encoder_fps $2"; shift 2 ;;
        --encoder_resolution)
            encoder_resolution="--encoder_resolution $2"; shift 2 ;;
        --distill_vision_layer)
            distill_vision_layer="--distill_vision_layer $2"; shift 2 ;;
        --distill_audio_layer)
            distill_audio_layer="--distill_audio_layer $2"; shift 2 ;;
        --lambda_distill_image_cosine)
            lambda_distill_image_cosine="--lambda_distill_image_cosine $2"; shift 2 ;;
        --lambda_distill_image_distance)
            lambda_distill_image_distance="--lambda_distill_image_distance $2"; shift 2 ;;
        --lambda_distill_video_cosine)
            lambda_distill_video_cosine="--lambda_distill_video_cosine $2"; shift 2 ;;
        --lambda_distill_video_distance)
            lambda_distill_video_distance="--lambda_distill_video_distance $2"; shift 2 ;;
        --lambda_distill_audio_t_axis)
            lambda_distill_audio_t_axis="--lambda_distill_audio_t_axis $2"; shift 2 ;;
        --lambda_distill_audio_d_axis)
            lambda_distill_audio_d_axis="--lambda_distill_audio_d_axis $2"; shift 2 ;;
        --lambda_group_distill)
            lambda_group_distill="--lambda_group_distill $2"; shift 2 ;;
        --distill_margin_cosine)
            distill_margin_cosine="--distill_margin_cosine $2"; shift 2 ;;
        --distill_margin_distance)
            distill_margin_distance="--distill_margin_distance $2"; shift 2 ;;
        --distill_w_hyper)
            distill_w_hyper="--distill_w_hyper $2"; shift 2 ;;
        --distill_audio_type)
            distill_audio_type="--distill_audio_type $2"; shift 2 ;;
        --distill_proj_type)
            distill_proj_type="--distill_proj_type $2"; shift 2 ;;
        --distill_proj_layers)
            distill_proj_layers="--distill_proj_layers $2"; shift 2 ;;
        --distill_proj_hidden_dim)
            distill_proj_hidden_dim="--distill_proj_hidden_dim $2"; shift 2 ;;
        --distill_use_conv3d)
            distill_use_conv3d="--distill_use_conv3d"; shift ;;
        --no_distill_use_conv3d)
            distill_use_conv3d="--no-distill_use_conv3d"; shift ;;
        --distill_proj_before_agg)
            distill_proj_before_agg="--distill_proj_before_agg"; shift ;;
        --no_distill_proj_before_agg)
            distill_proj_before_agg="--no-distill_proj_before_agg"; shift ;;
        --distill_dim_schedule)
            distill_dim_schedule="--distill_dim_schedule $2"; shift 2 ;;
        --distill_use_sampled)
            distill_use_sampled="--distill_use_sampled"; shift ;;
        --no_distill_use_sampled)
            distill_use_sampled="--no-distill_use_sampled"; shift ;;
        --distill_spatial_norm)
            distill_spatial_norm="--distill_spatial_norm"; shift ;;
        --no_distill_spatial_norm)
            distill_spatial_norm="--no-distill_spatial_norm"; shift ;;
        --distill_spatial_norm_gamma)
            distill_spatial_norm_gamma="--distill_spatial_norm_gamma $2"; shift 2 ;;
        --distill_use_dist_matrix)
            distill_use_dist_matrix="--distill_use_dist_matrix"; shift ;;
        --no_distill_use_dist_matrix)
            distill_use_dist_matrix="--no-distill_use_dist_matrix"; shift ;;
        --adaptive_distill_balance)
            adaptive_distill_balance="--adaptive_distill_balance"; shift ;;
        --no_adaptive_distill_balance)
            adaptive_distill_balance="--no-adaptive_distill_balance"; shift ;;
        --adaptive_distill_use_gradient)
            adaptive_distill_use_gradient="--adaptive_distill_use_gradient"; shift ;;
        --no_adaptive_distill_use_gradient)
            adaptive_distill_use_gradient="--no-adaptive_distill_use_gradient"; shift ;;
        --adaptive_distill_video_ratio)
            adaptive_distill_video_ratio="--adaptive_distill_video_ratio $2"; shift 2 ;;
        --adaptive_distill_audio_ratio)
            adaptive_distill_audio_ratio="--adaptive_distill_audio_ratio $2"; shift 2 ;;
        --distill_upload_mode)
            distill_upload_mode="--distill_upload_mode"; shift ;;
        --no_distill_upload_mode)
            distill_upload_mode="--no-distill_upload_mode"; shift ;;
        --distill_video_gpu_map)
            distill_video_gpu_map="--distill_video_gpu_map $2"; shift 2 ;;
        --distill_image_gpu_id)
            distill_image_gpu_id="--distill_image_gpu_id $2"; shift 2 ;;
        --distill_audio_gpu_id)
            distill_audio_gpu_id="--distill_audio_gpu_id $2"; shift 2 ;;
        --distill_num_upload_workers)
            distill_num_upload_workers="--distill_num_upload_workers $2"; shift 2 ;;
        --distill_processor_path)
            distill_processor_path="--distill_processor_path $2"; shift 2 ;;
        --qk_norm)
            qk_norm_flag="--qk_norm"; shift ;;
        --no_qk_norm)
            qk_norm_flag="--no-qk_norm"; shift ;;
        --contrastive_type)
            contrastive_type="--contrastive_type $2"; shift 2 ;;
        --contrastive_embed_dim)
            contrastive_embed_dim="--contrastive_embed_dim $2"; shift 2 ;;
        --contrastive_nhead)
            contrastive_nhead="--contrastive_nhead $2"; shift 2 ;;
        --self_attn_layers)
            self_attn_layers="--self_attn_layers $2"; shift 2 ;;
        --cross_attn_layers)
            cross_attn_layers="--cross_attn_layers $2"; shift 2 ;;
        --max_audio_tokens_per_seg)
            max_audio_tokens_per_seg="--max_audio_tokens_per_seg $2"; shift 2 ;;
        --max_spatial_h)
            max_spatial_h="--max_spatial_h $2"; shift 2 ;;
        --max_spatial_w)
            max_spatial_w="--max_spatial_w $2"; shift 2 ;;
        --contrastive_dim_feedforward)
            contrastive_dim_feedforward="--contrastive_dim_feedforward $2"; shift 2 ;;
        --contrastive_dropout)
            contrastive_dropout="--contrastive_dropout $2"; shift 2 ;;
        --warmup_steps)
            warmup_steps="--warmup_steps $2"; shift 2 ;;
        --max_steps)
            max_steps="--max_steps $2"; shift 2 ;;
        --pretrained_contrastive_checkpoint)
            pretrained_contrastive_checkpoint="--pretrained_contrastive_checkpoint $2"; shift 2 ;;
        --pretrained_disc_checkpoint)
            pretrained_disc_checkpoint="--pretrained_disc_checkpoint $2"; shift 2 ;;
        --pretrained_disc_load_optim)
            pretrained_disc_load_optim_flag="--pretrained_disc_load_optim"; shift ;;
        --no_pretrained_disc_load_optim|--no-pretrained_disc_load_optim)
            pretrained_disc_load_optim_flag="--no-pretrained_disc_load_optim"; shift ;;
        --video_loss_clamp)
            video_loss_clamp_flag="--video_loss_clamp"; shift ;;
        --no_video_loss_clamp)
            video_loss_clamp_flag="--no-video_loss_clamp"; shift ;;
        --video_recon_clamp_max)
            video_recon_clamp_max="--video_recon_clamp_max $2"; shift 2 ;;
        --video_lpips_clamp_max)
            video_lpips_clamp_max="--video_lpips_clamp_max $2"; shift 2 ;;
        --video_kl_clamp_max)
            video_kl_clamp_max="--video_kl_clamp_max $2"; shift 2 ;;
        --video_learn_logvar)
            video_learn_logvar_flag="--video_learn_logvar"; shift ;;
        --no_video_learn_logvar)
            video_learn_logvar_flag="--no-video_learn_logvar"; shift ;;
        --video_logvar_init)
            video_logvar_init="--video_logvar_init $2"; shift 2 ;;
        --gradient_accumulation_steps)
            gradient_accumulation_steps="--gradient_accumulation_steps $2"; shift 2 ;;
        --adaptive_loss_balance_v2)
            adaptive_loss_balance_v2="--adaptive_loss_balance_v2"; shift ;;
        --no_adaptive_loss_balance_v2)
            adaptive_loss_balance_v2="--no-adaptive_loss_balance_v2"; shift ;;
        --adaptive_anchor_source)
            adaptive_anchor_source="--adaptive_anchor_source $2"; shift 2 ;;
        --adaptive_anchor_ema_decay)
            adaptive_anchor_ema_decay="--adaptive_anchor_ema_decay $2"; shift 2 ;;
        --adaptive_anchor_warmup_steps)
            adaptive_anchor_warmup_steps="--adaptive_anchor_warmup_steps $2"; shift 2 ;;
        --adaptive_scale_clamp_min)
            adaptive_scale_clamp_min="--adaptive_scale_clamp_min $2"; shift 2 ;;
        --adaptive_scale_clamp_max)
            adaptive_scale_clamp_max="--adaptive_scale_clamp_max $2"; shift 2 ;;
        --adaptive_ratio_video)
            adaptive_ratio_video="--adaptive_ratio_video $2"; shift 2 ;;
        --adaptive_ratio_audio)
            adaptive_ratio_audio="--adaptive_ratio_audio $2"; shift 2 ;;
        --adaptive_ratio_contrastive)
            adaptive_ratio_contrastive="--adaptive_ratio_contrastive $2"; shift 2 ;;
        --adaptive_anchor_source_stage1)
            adaptive_anchor_source_stage1="--adaptive_anchor_source_stage1 $2"; shift 2 ;;
        --adaptive_ratio_video_stage1)
            adaptive_ratio_video_stage1="--adaptive_ratio_video_stage1 $2"; shift 2 ;;
        --adaptive_ratio_audio_stage1)
            adaptive_ratio_audio_stage1="--adaptive_ratio_audio_stage1 $2"; shift 2 ;;
        --adaptive_ratio_contrastive_stage1)
            adaptive_ratio_contrastive_stage1="--adaptive_ratio_contrastive_stage1 $2"; shift 2 ;;
        --adaptive_v2_stage2_use_gradient)
            adaptive_v2_stage2_use_gradient_flag="--adaptive_v2_stage2_use_gradient"; shift ;;
        --no_adaptive_v2_stage2_use_gradient)
            adaptive_v2_stage2_use_gradient_flag="--no-adaptive_v2_stage2_use_gradient"; shift ;;
        --adaptive_v2_stage2_blend_steps)
            adaptive_v2_stage2_blend_steps="--adaptive_v2_stage2_blend_steps $2"; shift 2 ;;
        --gradient_ratio_video_stage2)
            gradient_ratio_video_stage2="--gradient_ratio_video_stage2 $2"; shift 2 ;;
        --gradient_ratio_audio_stage2)
            gradient_ratio_audio_stage2="--gradient_ratio_audio_stage2 $2"; shift 2 ;;
        --freeze_video_vae)
            freeze_video_vae_flag="--freeze_video_vae"; shift ;;
        --no_freeze_video_vae)
            freeze_video_vae_flag="--no-freeze_video_vae"; shift ;;
        --freeze_video_vae_until_step)
            freeze_video_vae_until_step="--freeze_video_vae_until_step $2"; shift 2 ;;
        --freeze_audio_vae)
            freeze_audio_vae_flag="--freeze_audio_vae"; shift ;;
        --no_freeze_audio_vae)
            freeze_audio_vae_flag="--no-freeze_audio_vae"; shift ;;
        --freeze_audio_vae_until_step)
            freeze_audio_vae_until_step="--freeze_audio_vae_until_step $2"; shift 2 ;;
        --freeze_audio_encoder)
            freeze_audio_encoder_flag="--freeze_audio_encoder"; shift ;;
        --no_freeze_audio_encoder)
            freeze_audio_encoder_flag="--no-freeze_audio_encoder"; shift ;;
        --freeze_video_encoder)
            freeze_video_encoder_flag="--freeze_video_encoder"; shift ;;
        --no_freeze_video_encoder|--no-freeze_video_encoder)
            freeze_video_encoder_flag="--no-freeze_video_encoder"; shift ;;
        --contrastive_grad_scale_video)
            contrastive_grad_scale_video="--contrastive_grad_scale_video $2"; shift 2 ;;
        --contrastive_grad_scale_audio)
            contrastive_grad_scale_audio="--contrastive_grad_scale_audio $2"; shift 2 ;;
        --lr_video_vae)
            lr_video_vae="--lr_video_vae $2"; shift 2 ;;
        --lr_audio_vae)
            lr_audio_vae="--lr_audio_vae $2"; shift 2 ;;
        --lr_contrastive_head)
            lr_contrastive_head="--lr_contrastive_head $2"; shift 2 ;;
        --lr_distill_proj)
            lr_distill_proj="--lr_distill_proj $2"; shift 2 ;;
        --lr_video_logvar)
            lr_video_logvar="--lr_video_logvar $2"; shift 2 ;;
        --lr_video_vae_warmup_steps)
            lr_video_vae_warmup_steps="--lr_video_vae_warmup_steps $2"; shift 2 ;;
        --lr_video_vae_total_steps)
            lr_video_vae_total_steps="--lr_video_vae_total_steps $2"; shift 2 ;;
        --lr_video_vae_start_step)
            lr_video_vae_start_step="--lr_video_vae_start_step $2"; shift 2 ;;
        --lr_video_vae_min_ratio)
            lr_video_vae_min_ratio="--lr_video_vae_min_ratio $2"; shift 2 ;;
        --lr_audio_vae_warmup_steps)
            lr_audio_vae_warmup_steps="--lr_audio_vae_warmup_steps $2"; shift 2 ;;
        --lr_audio_vae_total_steps)
            lr_audio_vae_total_steps="--lr_audio_vae_total_steps $2"; shift 2 ;;
        --lr_audio_vae_start_step)
            lr_audio_vae_start_step="--lr_audio_vae_start_step $2"; shift 2 ;;
        --lr_audio_vae_min_ratio)
            lr_audio_vae_min_ratio="--lr_audio_vae_min_ratio $2"; shift 2 ;;
        --eval_steps)
            eval_steps="--eval_steps $2"; shift 2 ;;
        --save_steps)
            save_steps="--save_steps $2"; shift 2 ;;
        --use_audio_disc)
            use_audio_disc_flag="--use_audio_disc"; shift ;;
        --no_audio_disc)
            use_audio_disc_flag="--no-use_audio_disc"; shift ;;
        --audio_disc_start_step)
            audio_disc_start_step="--audio_disc_start_step $2"; shift 2 ;;
        --lambda_audio_adv)
            lambda_audio_adv="--lambda_audio_adv $2"; shift 2 ;;
        --lambda_audio_feature_matching)
            lambda_audio_feature_matching="--lambda_audio_feature_matching $2"; shift 2 ;;
        --lr_disc)
            lr_disc="--lr_disc $2"; shift 2 ;;
        --disc_max_grad_norm)
            disc_max_grad_norm="--disc_max_grad_norm $2"; shift 2 ;;
        --disc_dtype)
            disc_dtype="--disc_dtype $2"; shift 2 ;;
        --use_video_disc)
            use_video_disc_flag="--use_video_disc"; shift ;;
        --no_video_disc|--no-use_video_disc)
            use_video_disc_flag="--no-use_video_disc"; shift ;;
        --video_disc_start_step)
            video_disc_start_step="--video_disc_start_step $2"; shift 2 ;;
        --lambda_video_adv)
            lambda_video_adv="--lambda_video_adv $2"; shift 2 ;;
        --video_disc_loss_type)
            video_disc_loss_type="--video_disc_loss_type $2"; shift 2 ;;
        --video_disc_adaptive_weight)
            video_disc_adaptive_weight_flag="--video_disc_adaptive_weight"; shift ;;
        --no_video_disc_adaptive_weight|--no-video_disc_adaptive_weight)
            video_disc_adaptive_weight_flag="--no-video_disc_adaptive_weight"; shift ;;
        --video_disc_adaptive_weight_max)
            video_disc_adaptive_weight_max="--video_disc_adaptive_weight_max $2"; shift 2 ;;
        --video_disc_lazy_threshold)
            video_disc_lazy_threshold="--video_disc_lazy_threshold $2"; shift 2 ;;
        --distill_every_steps)
            distill_every_steps_flag="--distill_every_steps"; shift ;;
        --no_distill_every_steps|--no-distill_every_steps)
            distill_every_steps_flag="--no-distill_every_steps"; shift ;;
        *)
            if [ -z "${config}" ]; then
                config="$1"; shift
            else
                echo "Error: 未知参数 '$1'"; exit 1
            fi
            ;;
    esac
done

if [ -z "${config}" ]; then
    echo "Error: 请指定配置文件路径"
    echo "Usage: bash $0 <config_yaml> [--gpus 0,1,2,3] [--debug]"
    exit 1
fi

if [ ! -f "${config}" ]; then
    echo "Error: 配置文件不存在: ${config}"
    exit 1
fi

# ============================================================================
# 从 config 路径提取 tag
# ============================================================================
config_tag="${config#${REPO_ROOT}/}"
config_tag="${config_tag#./}"
config_tag="${config_tag#configs/}"
config_tag="${config_tag#/}"
config_tag="${config_tag%.yaml}"
tag="omnivae_${config_tag}"
tag_name="${tag//\//_}"
echo "tag: ${tag}"
# ============================================================================
# 输出目录（从 config 的 output.exp_root 读取，保持与 Python trainer 一致）
# ============================================================================
exp_root=$(python - "${config}" <<'PY'
import os
import re
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
repo_root = Path(os.environ.get("OMNIVAE_REPO_ROOT", os.getcwd())).resolve()
defaults = {
    "OMNIVAE_REPO_ROOT": str(repo_root),
    "OMNIVAE_CKPT_ROOT": str(repo_root / "ckpts"),
    "OMNIVAE_DATA_ROOT": str(repo_root / "data"),
    "OMNIVAE_EXP_ROOT": str(repo_root / "exp"),
}

def expand_known_vars(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, defaults.get(name, match.group(0)))

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, value)
    return os.path.expanduser(os.path.expandvars(value))

with config_path.open("r") as f:
    cfg = yaml.safe_load(f) or {}
exp_root = str((cfg.get("output") or {}).get("exp_root") or "$OMNIVAE_EXP_ROOT")
exp_root = expand_known_vars(exp_root)
path = Path(exp_root)
if not path.is_absolute():
    path = repo_root / path
print(path)
PY
)
exp_dir="${exp_root}/${tag}"
log_dir="${exp_dir}/log"

if [ "${dry_run}" != "1" ]; then
    mkdir -p "${log_dir}"
fi

# ============================================================================
# 分布式环境变量映射 & 模式检测
# ============================================================================
if [ -n "${PET_NNODES}" ] && [ "${PET_NNODES}" -gt 1 ] 2>/dev/null; then
    # ---- 分布式多机模式 ----
    dist_mode="distributed"

    MASTER_IP=""
    for i in $(seq 1 120); do
        MASTER_IP=$(getent hosts ${PET_MASTER_ADDR} | awk '{print $1}' | head -n 1)
        [ -n "${MASTER_IP}" ] && break
        echo "Waiting for DNS resolution of ${PET_MASTER_ADDR}... (${i}/60)"
        sleep 2
    done

    if [ -z "${MASTER_IP}" ]; then
        echo "Warning: getent failed after 60 retries, trying nslookup..."
        MASTER_IP=$(nslookup ${PET_MASTER_ADDR} 2>/dev/null | grep -A1 'Name:' | grep 'Address:' | awk '{print $2}')
    fi

    if [ -z "${MASTER_IP}" ]; then
        echo "Error: Cannot resolve MASTER_ADDR: ${PET_MASTER_ADDR}"
        exit 1
    fi

    echo "Resolved MASTER_ADDR: ${PET_MASTER_ADDR} -> ${MASTER_IP}"

    export MASTER_ADDR=${MASTER_IP}
    export MASTER_PORT=${PET_MASTER_PORT}

    NNODES=${PET_NNODES}
    NPROC_PER_NODE=${PET_NPROC_PER_NODE}
    NODE_RANK=${PET_NODE_RANK}
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

    num_gpus=${NPROC_PER_NODE}

    log_stdout="${log_dir}/${tag_name}_node${NODE_RANK}_stdout.log"
    log_stderr="${log_dir}/${tag_name}_node${NODE_RANK}_stderr.log"
else
    # ---- 单机模式 ----
    dist_mode="standalone"

    NNODES=1
    NODE_RANK=0
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}

    log_stdout="${log_dir}/${tag_name}_stdout.log"
    log_stderr="${log_dir}/${tag_name}_stderr.log"
fi

# ============================================================================
# GPU 配置
# ============================================================================
if [ ${debug_mode} -eq 1 ]; then
    dist_mode="standalone"
    NNODES=1
    NODE_RANK=0
    if [ -z "${gpu_ids}" ]; then
        gpu_ids="0"
    else
        gpu_ids=$(echo "${gpu_ids}" | cut -d',' -f1)
    fi
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    num_gpus=1
    NPROC_PER_NODE=1
    debug_cmd="--debug_ip ${debug_ip} --debug_port ${debug_port} --debug 1"
elif [ "${dist_mode}" = "standalone" ]; then
    if [ "${dry_run}" = "1" ]; then
        num_gpus=1
    elif [ -n "${gpu_ids}" ]; then
        export CUDA_VISIBLE_DEVICES="${gpu_ids}"
        num_gpus=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "all" ]; then
        num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l)
    else
        num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    fi
    if [ "${num_gpus}" -eq 0 ]; then
        echo "Error: 未检测到可用 GPU"
        exit 1
    fi
    NPROC_PER_NODE=${num_gpus}
    debug_cmd=""
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============================================================================
# 环境变量
# ============================================================================
export PYTHONWARNINGS="default"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_IB_TIMEOUT=30
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=0

python_scripts="omnivae/train/train_audio_video_vae.py"

# ============================================================================
# 打印配置
# ============================================================================
echo "============================================"
if [ "${dist_mode}" = "distributed" ]; then
    echo "  分布式多机多卡训练"
else
    echo "  单机多卡训练"
fi
echo "============================================"
echo "Mode         : ${dist_mode}"
echo "Config       : ${config}"
echo "Base Tag     : ${tag}  (Python will append detail suffix)"
echo "Exp Root     : ${exp_dir}"
echo "NNODES       : ${NNODES}"
echo "NPROC/NODE   : ${NPROC_PER_NODE}"
echo "NODE_RANK    : ${NODE_RANK}"
echo "WORLD_SIZE   : ${WORLD_SIZE}"
echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "GPUs(local)  : ${num_gpus}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
echo "Debug        : ${debug_mode}"
echo "Dry run      : ${dry_run}"
echo "Continue     : ${continue_train:-disabled}"
echo "Valid only   : ${valid_only_flag:-disabled}"
echo "Checkpoint   : ${checkpoint_flag:-auto (latest)}"
echo "Max steps    : ${max_steps:-config default}"
echo "Loss overrides: ${loss_override_flags:-none (use config)}"
echo "============================================"

# ============================================================================
# Loss 控制 flags
# ============================================================================
loss_flags="${loss_override_flags} ${lambda_group_video} ${lambda_group_audio} ${lambda_group_contrastive} ${lambda_segment_contrastive} ${lambda_global_contrastive} ${lambda_video_kl} ${lambda_video_lpips} ${lambda_audio_kl} ${spatial_pool_mode} ${spatial_merge_factor} ${segment_count} ${num_negatives} ${num_negative_videos} ${same_long_video_priority_flag} ${same_long_video_num_negatives} ${num_negatives_with_sibling} ${num_negatives_no_sibling} ${segment_temporal_pool_mode} ${global_temporal_pool_mode} ${contrastive_transformer_layers} ${transformer_nhead} ${spatial_transformer_layers} ${segment_transformer_layers} ${global_transformer_layers} ${contrastive_module_size} ${spatial_module_size} ${segment_module_size} ${global_module_size} ${cnn_num_blocks_per_stage} ${cnn_kernel_size} ${use_sdpa_flag} ${lr} ${batch_size} ${use_ema_flag} ${contrastive_use_mean_flag} ${valid_only_flag} ${checkpoint_flag} ${val_segment_num_negatives} ${val_segment_num_negative_videos} ${val_global_num_negatives} ${val_contrastive_max_samples} ${eval_video_recon_flag} ${eval_audio_recon_flag} ${eval_contrastive_flag} ${eval_contrastive_in_all_flag} ${exp_name_suffix} ${pretrained_checkpoint} ${pretrained_video_checkpoint} ${pretrained_audio_checkpoint} ${keep_audio_vae_pretrained_flag} ${global_contrastive_start_steps} ${video_distill_start_step} ${audio_distill_start_step} ${segment_avclip_start_steps} ${segment_count_weights} ${freeze_vae_encoders_flag} ${spatial_transform_mode} ${spatial_roundtrip_short_edge} ${train_metadata_path} ${grad_log_steps} ${adaptive_loss_balance} ${adaptive_balance_audio_ratio} ${adaptive_balance_contrastive_ratio} ${adaptive_loss_balance_by_uncertainty} ${uncertainty_warmup_steps} ${adaptive_loss_balance_by_gradient} ${gradient_balance_video_ratio} ${gradient_balance_audio_ratio} ${gradient_balance_clamp_max} ${gradient_balance_interval} ${dtype} ${video_vae_dtype} ${audio_vae_dtype} ${contrastive_dtype} ${num_frames} ${gradient_checkpointing} ${max_grad_norm} ${use_semantic_distill_flag} ${semantic_model_path} ${semantic_api_url} ${encoder_fps} ${encoder_resolution} ${distill_vision_layer} ${distill_audio_layer} ${lambda_distill_image_cosine} ${lambda_distill_image_distance} ${lambda_distill_video_cosine} ${lambda_distill_video_distance} ${lambda_distill_audio_t_axis} ${lambda_distill_audio_d_axis} ${lambda_group_distill} ${distill_margin_cosine} ${distill_margin_distance} ${distill_w_hyper} ${distill_audio_type} ${distill_proj_type} ${distill_proj_layers} ${distill_proj_hidden_dim} ${distill_use_conv3d} ${distill_proj_before_agg} ${distill_dim_schedule} ${distill_use_sampled} ${distill_spatial_norm} ${distill_spatial_norm_gamma} ${distill_use_dist_matrix} ${adaptive_distill_balance} ${adaptive_distill_use_gradient} ${adaptive_distill_video_ratio} ${adaptive_distill_audio_ratio} ${distill_upload_mode} ${distill_video_gpu_map} ${distill_image_gpu_id} ${distill_audio_gpu_id} ${distill_num_upload_workers} ${distill_processor_path} ${qk_norm_flag} ${contrastive_type} ${contrastive_embed_dim} ${contrastive_nhead} ${self_attn_layers} ${cross_attn_layers} ${max_audio_tokens_per_seg} ${max_spatial_h} ${max_spatial_w} ${contrastive_dim_feedforward} ${contrastive_dropout} ${warmup_steps} ${max_steps} ${pretrained_contrastive_checkpoint} ${pretrained_disc_checkpoint} ${pretrained_disc_load_optim_flag} ${video_loss_clamp_flag} ${video_recon_clamp_max} ${video_lpips_clamp_max} ${video_kl_clamp_max} ${video_learn_logvar_flag} ${video_logvar_init} ${gradient_accumulation_steps} ${adaptive_loss_balance_v2} ${adaptive_anchor_source} ${adaptive_anchor_ema_decay} ${adaptive_anchor_warmup_steps} ${adaptive_scale_clamp_min} ${adaptive_scale_clamp_max} ${adaptive_ratio_video} ${adaptive_ratio_audio} ${adaptive_ratio_contrastive} ${freeze_video_vae_flag} ${freeze_video_vae_until_step} ${contrastive_grad_scale_video} ${contrastive_grad_scale_audio} ${lr_video_vae} ${lr_audio_vae} ${lr_contrastive_head} ${lr_distill_proj} ${lr_video_logvar} ${video_model_name} ${video_model_config} ${pretrained_video_model_path} ${adaptive_anchor_source_stage1} ${adaptive_ratio_video_stage1} ${adaptive_ratio_audio_stage1} ${adaptive_ratio_contrastive_stage1} ${adaptive_v2_stage2_use_gradient_flag} ${adaptive_v2_stage2_blend_steps} ${gradient_ratio_video_stage2} ${gradient_ratio_audio_stage2} ${lr_video_vae_warmup_steps} ${lr_video_vae_total_steps} ${lr_video_vae_start_step} ${lr_video_vae_min_ratio} ${freeze_audio_vae_flag} ${freeze_audio_vae_until_step} ${freeze_audio_encoder_flag} ${freeze_video_encoder_flag} ${lr_audio_vae_warmup_steps} ${lr_audio_vae_total_steps} ${lr_audio_vae_start_step} ${lr_audio_vae_min_ratio} ${eval_steps} ${save_steps} ${use_audio_disc_flag} ${audio_disc_start_step} ${lambda_audio_adv} ${lambda_audio_feature_matching} ${lr_disc} ${disc_max_grad_norm} ${disc_dtype} ${use_video_disc_flag} ${video_disc_start_step} ${lambda_video_adv} ${video_disc_loss_type} ${video_disc_adaptive_weight_flag} ${video_disc_adaptive_weight_max} ${video_disc_lazy_threshold} ${distill_every_steps_flag} ${exp_name_arg} ${reset_scheduler_on_resume_flag}"

# ============================================================================
# 启动训练 (torchrun 分布式模式)
# ============================================================================
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

cmd="${TORCHRUN_BIN} \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${python_scripts} \
    --config ${config} \
    --tag ${tag} \
    ${continue_train} \
    ${loss_flags} \
    ${debug_cmd}"

echo "Executing: ${cmd}"
if [ "${dry_run}" = "1" ]; then
    echo "OMNIVAE_DRY_RUN=1, command was not executed."
    exit 0
fi
eval ${cmd} 2> >(tee -a "${log_stderr}" >&2) | tee -a "${log_stdout}"

echo "Training completed!"
