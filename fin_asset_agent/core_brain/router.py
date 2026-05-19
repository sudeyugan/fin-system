# core_brain/router.py
from pydantic import BaseModel
from typing import Literal

class IntentResponse(BaseModel):
    intent: Literal["QA_RAG", "ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "CHIT_CHAT"]
    extracted_entities: dict

class CoreRouter:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def route_query(self, user_query: str) -> IntentResponse:
        """
        利用轻量级高响应大模型解析意图，避免复杂路由带来延迟。
        例如："帮我分析我的持仓，如何降低波动率？" -> PORTFOLIO_REVIEW
        """
        prompt = f"Analyze the intent of the following financial query: {user_query}"
        # 伪代码：调用大模型返回 JSON 格式结果
        return self.llm.generate_structured(prompt, response_model=IntentResponse)