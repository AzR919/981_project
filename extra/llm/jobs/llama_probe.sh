#!/bin/bash
# Wait for the user's Hugging Face token on the lab home (~/.hf_token) and for
# access to the gated Llama models, then put the Llama probes at the front of
# the unstarted queue. Run from the queue worktree (~/rlvr-queue).
while true; do
  OK=$(ssh -n -o ConnectTimeout=20 secb1010-a01 '[ -s ~/.hf_token ] && HF_TOKEN=$(cat ~/.hf_token) \
    /tmp/asa715/llmenv/bin/python -c "
from huggingface_hub import model_info
for m in (\"meta-llama/Llama-3.2-1B-Instruct\", \"meta-llama/Llama-3.2-3B-Instruct\", \"meta-llama/Llama-3.1-8B-Instruct\"):
    model_info(m)
print(\"yes\")" 2>/dev/null')
  [ "$OK" = yes ] && break
  sleep 300
done
N=$(cat queue.txt.started)
{ head -n "$N" queue.txt; cat extra/llm/jobs/probe_llama.txt; tail -n +"$((N + 1))" queue.txt; } > queue.tmp && mv queue.tmp queue.txt
echo "$(date +%H:%M) token and Llama access found; Llama probes queued" >> llama_probe.log
