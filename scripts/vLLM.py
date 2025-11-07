# vLLM usage when using quantized model

from vllm import LLM
from transformers import AutoTokenizer

login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'ibm-granite/granite-4.0-1b'

# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

engine = LLM(
    model=model_id,
    max_num_seqs=8,
    max_num_batched_tokens=512,
    dtype="float16",
    gpu_memory_utilization=0.9,
    enforce_eager=False,
    enable_chunked_prefill=True
)

prompt = 'Once upon a time'

result = engine.generate(prompt, max_tokens=256)
print(result)