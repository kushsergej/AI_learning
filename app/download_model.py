from transformers import AutoTokenizer, AutoModelForCausalLM



# load model from HuggingFace
try:
    model_id = 'ibm-granite/granite-3.3-2b-instruct'
    model_cache = 'model_cache'

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=model_cache)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', cache_dir=model_cache)

    tokenizer.save_pretrained(model_cache)
    model.save_pretrained(model_cache)
    print(f'✅ Model {model_id} loaded successfully to {model_cache}')
except Exception as e:
    print(f'❌ Error loading model: {e}')