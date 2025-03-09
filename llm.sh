# Serving LLM
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
# vllm serve "Qwen/Qwen2.5-32B-Instruct" --dtype auto --api-key token-abc123 --tensor-parallel-size 4
# CUDA_VISIBLE_DEVICES=2,3 vllm serve "Qwen/Qwen2.5-32B-Instruct" --dtype auto --tensor-parallel-size 2
# CUDA_VISIBLE_DEVICES=2,3 vllm serve "Qwen/Qwen2.5-32B-Instruct" --dtype auto --tensor-parallel-size 2
# vllm serve "Qwen/Qwen2.5-32B" --dtype auto --tensor-parallel-size 2
# vllm serve "Qwen/Qwen2.5-14B" --dtype auto --tensor-parallel-size 1 --max_model_len 8192
vllm serve "Qwen/Qwen2.5-14B" --dtype auto --tensor-parallel-size 1 --max_model_len 8192 --gpu-memory-utilization 0.7