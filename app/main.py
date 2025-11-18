import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# load model from HuggingFace
try:
    model_id = 'ibm-granite/granite-3.3-2b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
    print(f'✅ Model {model_id} loaded successfully')
except Exception as e:
    pipe = None
    print(f'❌ Error loading model: {e}')



# logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



# FastAPI
app = FastAPI()

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'Response status: {response.status_code}')
    return response



# LLM class
class LLM_Request(BaseModel):
    llm_prompt: str

class LLM_Response(BaseModel):
    llm_response: str



# FastAPI endpoints
@app.get('/')
async def healthcheck() -> Response:
    if pipe:
        return JSONResponse(status_code=200, content={'status': 'Healthy'})
    return JSONResponse(status_code=500, content={'error': str(e)})



@app.post('/llm', response_model=LLM_Response)
async def llm_response(llm_request: LLM_Request):
    if not pipe:
        return JSONResponse(status_code=500, content={'error': str(e)})
    try:
        logger.info(f'Initial prompt: {llm_request.llm_prompt}')

        result = result = pipe(
            llm_request.llm_prompt,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=256,
            do_sample=True,
            return_full_text=False
        )
        return LLM_Response(llm_response=f'{result[0]['generated_text']} [from {model_id}]')
    except Exception as e:
        return {'err message': str(e)}



if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='localhost',
        port=8000
        )



# ------------------------------------- #
# docker build -t fastapi-llm ./app
# docker run -d -p 8000:8000 --name myapp fastapi-llm

#  curl --silent --request POST --header 'Content-Type: application/json' --data '{"llm_prompt": "Who is the Rome Pope now?"}' http://localhost:8000/llm