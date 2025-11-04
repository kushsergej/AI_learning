# QLoRA fine-tuning

import hf_xet
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

try:
    print('>>> Load LLM')
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        fan_in_fan_out=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    print('>>> Tokenize loaded datasets')
    dataset = load_dataset('mteb/tweet_sentiment_extraction')
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir='./LoRA_output',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=100,
        seed=42,
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=tokenized_datasets['train']
    )

    print('>>> Training')
    trainer.train()

    print('>>> Validating')
    trainer.evaluate()
except Exception as e:
    print(e)