# bitsandbytes quantization

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import bitsandbytes

load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'PrunaAI/mistralai-Mistral-7B-Instruct-v0.2-bnb-4bit-smashed'

# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')

prompt = '''
<|system|> You are a experienced DevOps Engineer with robust skills in Azure and Python </s>\
<|user|> How to write a unit tests on Python project? </s>\
<|assistant|>
'''

pipe = pipeline(task='text-generation', model=model_4bit, tokenizer=tokenizer)
result = pipe(
    prompt,
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True,
    return_full_text=False
)
print(f'{[model_8bit.get_memory_footprint()]} >>> {result[0]['generated_text']}')