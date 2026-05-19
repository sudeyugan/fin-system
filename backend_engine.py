# backend_engine.py
import os
import json
import numpy as np
from scipy.optimize import minimize
from openai import OpenAI

# 1. 确定性计算沙箱
class MVOSolver:
    @staticmethod
    def optimize_portfolio(expected_returns: list, cov_matrix: list) -> list:
        """执行马科维茨均值-方差优化 (MVO)，求解最小方差组合"""
        rets = np.array(expected_returns)
        cov = np.array(cov_matrix)
        num_assets = len(rets)
        
        # 目标函数：最小化组合方差
        def portfolio_variance(weights):
            return weights.T @ cov @ weights

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 权重和为 1
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))          # 禁止做空
        init_guess = np.ones(num_assets) / num_assets

        result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            return init_guess.tolist() # 失败则平分权重
        return result.x.tolist()

# 2. DeepSeek 驱动的 FinRL-X 工作流
class FinAgentWorkflow:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        # 统一使用 OpenAI SDK 兼容格式调用 DeepSeek
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "deepseek-chat" # 基础流程采用高响应的 DeepSeek-V3

    def run_workflow(self, user_query: str) -> dict:
        """运行完整控制流：S_t -> A_t (Sandbox) -> T_t -> R_t"""
        
        # --- 阶段 1: 资产筛选 (S_t) ---
        # 让 DeepSeek 扮演 Analyst Agent，将自然语言转化为结构化资产和模拟的马科维茨输入数据
        prompt_s = f"""
        你是一个资深金融分析师 Agent (S_t)。请根据用户的投资诉求："{user_query}"，从市场中筛选出最贴切的 3 个资产。
        必须严格以 JSON 格式返回，不要包含任何 Markdown 标记或外层包裹。格式如下：
        {{"assets": ["资产1", "资产2", "资产3"], "expected_returns": [0.12, 0.08, 0.15], "cov_matrix": [[0.04, 0.01, 0.02], [0.01, 0.03, 0.015], [0.02, 0.015, 0.06]]}}
        """
        response_s = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_s}],
            response_format={"type": "json_object"}
        )
        data_s = json.loads(response_s.choices[0].message.content)
        
        assets = data_s["assets"]
        expected_returns = data_s["expected_returns"]
        cov_matrix = data_s["cov_matrix"]

        # --- 阶段 2: 资产配置确定性计算 (A_t) ---
        # 严格脱离大模型，交由 Python 沙箱计算 base 权重
        base_weights = MVOSolver.optimize_portfolio(expected_returns, cov_matrix)

        # --- 阶段 3: 情绪/择时微调 (T_t) ---
        # 扮演 Portfolio Manager Agent，根据宏观情绪对沙箱权重进行微调
        prompt_t = f"""
        你是一个资产配置经理 Agent (T_t)。目前沙箱计算出的基础权重为：{dict(zip(assets, base_weights))}。
        请结合当前全球宏观政策或行业情绪，对这三个资产的权重进行微调（增配或减配，调整幅度控制在正负5%以内）。
        必须严格保持调整后的总权重之和为 1.0。
        请严格以 JSON 格式返回，不要包含任何 Markdown 标记。格式如下：
        {{"adjusted_weights": [权重1, 权重2, 权重3], "reason": "微调的宏观/情绪逻辑阐述"}}
        """
        response_t = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_t}],
            response_format={"type": "json_object"}
        )
        data_t = json.loads(response_t.choices[0].message.content)
        adjusted_weights = data_t["adjusted_weights"]
        timing_reason = data_t["reason"]

        # --- 阶段 4: 风险控制叠加 (R_t) ---
        # 扮演 Risk & Compliance Agent，执行合规性终审，阻断带有“保本”等绝对化用词
        prompt_r = f"""
        你是一个风控合规官 Agent (R_t)。请对以下配置方案进行合规审计：
        资产: {assets}
        最终权重: {adjusted_weights}
        投资策略理由: {timing_reason}
        
        审核要求：
        1. 检查总权重是否为 1.0。
        2. 文本中绝对不能出现“保本”、“稳赚不赔”、“零风险”等非法承诺。
        必须严格以 JSON 格式返回，格式如下：
        {{"status": "PASS" 或 "REJECT", "audit_report": "合规审计意见，若通过则给出专业的风控提示"}}
        """
        response_r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_r}],
            response_format={"type": "json_object"}
        )
        data_r = json.loads(response_r.choices[0].message.content)

        return {
            "assets": assets,
            "base_weights": base_weights,
            "final_weights": adjusted_weights if data_r["status"] == "PASS" else base_weights,
            "timing_reason": timing_reason,
            "risk_status": data_r["status"],
            "risk_report": data_r["audit_report"]
        }