#!/bin/bash
# Build the GPU environment for the LLM experiments on one lab machine.
# usage (on the lab machine):  bash llm/setup_env.sh
#
# Everything heavy lives in /tmp/asa715 (local disk, wiped at the 03:00
# reboot), so rerun this after a reboot; with the uv cache warm it takes a
# few minutes. Only uv itself (~30 MB) goes in the network home.
#   /tmp/asa715/llmenv   Python 3.12 venv with vLLM, torch, transformers, peft
#   /tmp/asa715/hf       Hugging Face cache (model weights, datasets)
#   /tmp/asa715/uv-cache uv download cache
# The exact versions installed are written to llmenv/freeze.txt and should be
# copied into each result folder.
set -e
ROOT=/tmp/asa715
mkdir -p $ROOT
export UV_CACHE_DIR=$ROOT/uv-cache
export PATH=$HOME/.local/bin:$PATH

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -x $ROOT/llmenv/bin/python ]; then
  uv venv --python 3.12 $ROOT/llmenv
fi
source $ROOT/llmenv/bin/activate
# vLLM pins a compatible torch; the rest follow.
uv pip install vllm transformers peft datasets accelerate bitsandbytes pandas scipy
uv pip freeze > $ROOT/llmenv/freeze.txt

python - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__, "transformers", transformers.__version__)
EOF
echo "ok: source $ROOT/llmenv/bin/activate; export HF_HOME=$ROOT/hf"
