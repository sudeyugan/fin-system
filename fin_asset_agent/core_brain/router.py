# core_brain/router.py
from pydantic import BaseModel
from typing import Literal
import json
from openai import OpenAI

class IntentResponse(BaseModel):
    intent: Literal["QA_RAG", "ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "CHIT_CHAT"]
    extracted_entities: dict

class CoreRouter:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
    def route_query(self, user_query: str) -> IntentResponse:
        prompt = f"分析此金融用户提问的意图并提取关键实体(如时间、资产名): {user_query}。要求返回 JSON，包含 'intent' 和 'extracted_entities'"
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个高效的路由器。必须返回合法的 JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={'type': 'json_object'}
        )
        data = json.loads(response.choices[0].message.content)
        return IntentResponse(**data)