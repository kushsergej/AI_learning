# Hugginface playground

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'ibm-granite/granite-3.3-2b-instruct'


# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')


prompt = [{'role': 'user', 'content': 'Pls give me a brief explanation of gravity in simple terms'}]
# print(f'tokens: {tokenizer.tokenize(prompt)}')
# print(f'ids: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))}')


pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
result = pipe(
    prompt,
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True,
    return_full_text=False
)
print(result[0]['generated_text'])