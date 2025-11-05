# Embeddings

import os
import hf_xet
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer


load_dotenv()
login(token=os.getenv('HF_READ_TOKEN'))
model_id = 'deepseek-ai/deepseek-coder-1.3b-instruct'


# ls -la /c/Users/Siarhei_Kushniaruk/.cache/huggingface/hub
model = SentenceTransformer(model_id)


sentences = [
    'King and Queen',
    'Man and woman',
    'Hubby and wifey'
]
for sentence in sentences:
    print(f'>>> {sentence}: {model.encode(sentence)}')