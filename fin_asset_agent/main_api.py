# main_api.py
import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core_brain.workflow import run_fin_agent_pipeline
from core_brain.router import CoreRouter

app = FastAPI()

# 允许跨域请求（前端 HTML 才能顺利访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AllocationRequest(BaseModel):
    query: str
    api_key: str

@app.post("/api/allocate")
async def allocate_portfolio(req: AllocationRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    try:
        # 调用之前写好的 DeepSeek + Sandbox 工作流
        result = run_fin_agent_pipeline(req.query, req.api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)