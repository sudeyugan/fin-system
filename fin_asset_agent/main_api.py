# main_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend_engine import FinAgentWorkflow  

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
        engine = FinAgentWorkflow(api_key=req.api_key)
        result = engine.run_workflow(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)