import logging
from typing import Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn



# logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()



# LLM class
class LLM_Request(BaseModel):
    llm_prompt: str
    llm_max_tokens: Optional[int] = 100

class LLM_Response(BaseModel):
    llm_response: str
    llm_tokens_used: int

def call_local_llm(prompt: str, max_tokens: int) -> str:
    return f'Mock LLM response for: {prompt} (max_tokens: {max_tokens})'




# FastAPI
app = FastAPI()

@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    logger.info(f'Incoming request: {request.method} {request.url}')
    response = await call_next(request)
    logger.info(f'Response status: {response.status_code}')
    return response



# FastAPI endpoints
@app.get('/')
async def health() -> Response:
    try:
        return JSONResponse(status_code=200, content={'status': 'Healthy'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})



@app.post('/llm', response_model=LLM_Response)
async def llm_response(llm_request: LLM_Request):
    try:
        logger.info(f'Prompt: {llm_request.llm_prompt}, max tokens: {llm_request.llm_max_tokens}')

        result = call_local_llm(llm_request.llm_prompt, llm_request.llm_max_tokens)
        return LLM_Response(
            llm_response=result,
            llm_tokens_used=len(result.split())
        )
    except Exception as e:
        return {'err message': str(e)}



if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='localhost',
        port=1234
        )