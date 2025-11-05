# QLoRA fine-tuning

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, TrainingArguments, Trainer
import torch
import bitsandbytes
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'PrunaAI/mistralai-Mistral-7B-Instruct-v0.2-bnb-4bit-smashed'

# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, trust_remote_code=True, device_map='auto')

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['query_key_value'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

tokenizer.pad_token = tokenizer.eos_token
model_4bit = prepare_model_for_kbit_training(model_4bit)
model_4bit = get_peft_model(model_4bit, config)

dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')
dataset = dataset.map(lambda samples: tokenizer(samples['quote']), batched=True)

trainer = transformers.Trainer(
    model=model_4bit,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=4,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=4,
        logging_steps=10,
        output_dir='QLoRA_outputs',
        save_strategy='epoch'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)








model_4bit.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained('QLoRA_outputs')

lora_config = LoraConfig.from_pretrained('QLoRA_outputs')
model_4bit = get_peft_model(model_4bit, lora_config)

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