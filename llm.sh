# Serving LLM
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3
vllm serve "Qwen/Qwen2.5-14B" --dtype auto --tensor-parallel-size 2 --max_model_len 8192 --gpu-memory-utilization 0.5
