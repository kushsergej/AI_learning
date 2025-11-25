from huggingface_hub import snapshot_download


model_id = 'ibm-granite/granite-3.3-2b-instruct'
model_path = 'app/model_snapshot'


# load model from HuggingFace
try:
    print(f'✅ Model {model_id} download started...')

    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False,       # critical for Docker: explicit files instead of symlinks
        ignore_patterns=['*.pt', '*.bin'],  # optimize storage: Exclude redundant pytorch weights if safetensors exist
        force_download=False                # ensures latest files overwrite any existing ones
    )

    print(f'✅ Model {model_id} downloaded successfully to {model_path}')
except Exception as e:
    print(f'❌ Error loading model: {e}')