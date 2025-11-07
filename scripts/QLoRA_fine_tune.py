# QLoRA fine-tuning

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, Trainer
import torch
# import bitsandbytes
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset


load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'meta-llama/Llama-3.2-1B-Instruct'


# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=quantization_config,
    device_map='auto'
)


dataset = load_dataset('Amod/mental_health_counseling_conversations', split='train')
# convert dataset from [Context, Response] to [Context, Response, tokenized_Context]
dataset = dataset.map(lambda sample: tokenizer(sample['Context']), batched=True)
# convert dataset from [Context, Response] to [Response, tokenized_Context]
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'Response'])


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['query_key_value'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
# model_4bit = prepare_model_for_kbit_training(model_4bit)
# model_4bit = get_peft_model(model_4bit, lora_config)


tokenizer.pad_token = tokenizer.eos_token
trainer = Trainer(
    model=model_4bit,
    train_dataset=dataset,
    args=TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=1,
        learning_rate=2e-4,
        save_total_limit=2,
        logging_steps=10,
        output_dir='QLoRA_outputs',
        overwrite_output_dir=True,
        save_strategy='epoch'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()


model_to_save = 'QLoRA_outputs/checkpoint_final'
trainer.save_model(model_to_save)
tokenizer.save_pretrained(model_to_save)
# lora_config = LoraConfig.from_pretrained('QLoRA_outputs')
# model_4bit = get_peft_model(model_4bit, lora_config)


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
print(f'{[model_4bit.get_memory_footprint()]} >>> {result[0]['generated_text']}')