#!/usr/bin/env bash
# Create symlinks from MOVA's hard-coded checkpoint paths to the existing files
# in Verse-Bench/models/. MOVA's eval_lip_sync / eval_audio_is_clap read fixed
# paths inside generation/evaluation/...; if you already downloaded the
# weights via setup_verse_bench.sh / download_weights.sh into Verse-Bench, this
# script makes them visible under MOVA's path too -- no copy, no disk overhead.
#
# Idempotent: existing symlinks pointing to the right target are left alone;
# existing real files at the destination are kept (and we just log a notice).
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_eval_dir="$(cd "$(dirname "${script_path}")" && pwd)"
project_root="$(cd "${my_eval_dir}/../../../.." && pwd)"

VERSE_BENCH_ROOT="${MY_EVAL_VERSE_BENCH_ROOT:-${project_root}/generation/evaluation/verse_bench}"
if [[ -f "${VERSE_BENCH_ROOT}/scripts/common.sh" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "${VERSE_BENCH_ROOT}/scripts/common.sh"
fi

MODELS_PATH_R="${MODELS_PATH:-${VERSE_BENCH_ROOT}/models}"
EVAL_ROOT="${project_root}/generation/evaluation"

# pairs: SOURCE_FILE  ->  DEST_SYMLINK
#
# Note for roberta-base: MOVA's git tracks the tokenizer files but not the
# weights, so the destination dir already exists. We link the single missing
# *.safetensors file instead of replacing the whole directory.
declare -a PAIRS=(
    "${MODELS_PATH_R}/syncnet_v2.model"
    "${EVAL_ROOT}/models/wav2lip/evaluation/syncnet_python/data/syncnet_v2.model"

    "${MODELS_PATH_R}/630k-audioset-fusion-best.pt"
    "${EVAL_ROOT}/metrics/audio_is_clap/clap_ckpt/630k-audioset-fusion-best.pt"

    "${MODELS_PATH_R}/roberta-base/model.safetensors"
    "${EVAL_ROOT}/models/roberta-base/model.safetensors"
)

echo "============================================"
echo "  link_canonical_paths.sh"
echo "============================================"
echo "MODELS_PATH   : ${MODELS_PATH_R}"
echo "EVAL_ROOT     : ${EVAL_ROOT}"
echo "============================================"

linked=0
skipped_existing=0
errors=0

i=0
while (( i < ${#PAIRS[@]} )); do
    src="${PAIRS[$i]}"; dst="${PAIRS[$((i+1))]}"
    i=$((i+2))

    if [[ ! -e "${src}" ]]; then
        echo ""
        echo "  [skip] source missing: ${src}"
        echo "         (download_weights.sh or setup_verse_bench.sh should have produced this)"
        errors=$((errors+1))
        continue
    fi

    # If dest already exists as a real file/dir (not a symlink), leave it alone.
    if [[ -e "${dst}" && ! -L "${dst}" ]]; then
        echo ""
        echo "  [keep] dest already exists (not a symlink): ${dst}"
        echo "         leaving as-is. Remove it manually if you want to point to ${src}."
        skipped_existing=$((skipped_existing+1))
        continue
    fi

    # If a symlink already points at the right target, no-op.
    if [[ -L "${dst}" ]]; then
        cur_target="$(readlink -f "${dst}")"
        if [[ "${cur_target}" == "$(readlink -f "${src}")" ]]; then
            echo ""
            echo "  [ok]   already linked: ${dst}"
            echo "         -> ${cur_target}"
            continue
        fi
        # Wrong target: rewrite.
        echo ""
        echo "  [redir] existing symlink points elsewhere, rewriting:"
        echo "          ${dst} -> ${cur_target}"
        rm "${dst}"
    fi

    mkdir -p "$(dirname "${dst}")"
    ln -s "${src}" "${dst}"
    echo ""
    echo "  [link] ${dst}"
    echo "         -> ${src}"
    linked=$((linked+1))
done

echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
echo "  new symlinks created : ${linked}"
echo "  already correct      : $((${#PAIRS[@]} / 2 - linked - skipped_existing - errors))"
echo "  kept (already file)  : ${skipped_existing}"
echo "  sources missing      : ${errors}"
echo ""
if (( errors > 0 )); then
    echo "Some source files are missing. Run:"
    echo "  bash generation/infer/t2av/my_eval/download_weights.sh"
    exit 1
fi
echo "Done. Re-run check_weights.sh to confirm everything resolves."
