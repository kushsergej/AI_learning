from transformers import AutoTokenizer, AutoModelForCausalLM



# load model from HuggingFace
try:
    model_id = 'ibm-granite/granite-3.3-2b-instruct'
    model_path = './model_cache'

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=model_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', cache_dir=model_path)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    print(f'✅ Model {model_id} loaded successfully to {model_path}')
except Exception as e:
    print(f'❌ Error loading model: {e}')