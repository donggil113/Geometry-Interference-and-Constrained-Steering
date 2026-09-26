#!/bin/bash
# Off-critical-path jobs, single-threaded, sequential (realizability + feature stats for all settings).
set -x
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
. .venv/bin/activate
python -u scripts/05_realizability.py gpt2_relu_jb_L6 48
python -u scripts/02_feature_stats.py gpt2_topk_oai_L6
python -u scripts/05_realizability.py gpt2_topk_oai_L6 48
python -u scripts/02_feature_stats.py gemma3_270m_jumprelu_L12
python -u scripts/05_realizability.py gemma3_270m_jumprelu_L12 48
echo CHAIN_DONE
