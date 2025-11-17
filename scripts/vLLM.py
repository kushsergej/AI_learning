# vLLM usage when using quantized model

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'ibm-granite/granite-4.0-1b'


# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)


prompts = [
    'Hello, my name is',
    'The president of the United States is',
]


guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params, temperature=0.2, top_p=0.9)
engine = LLM(model=model_id)
result = engine.generate(prompt, sampling_params, max_tokens=256)
print(result)