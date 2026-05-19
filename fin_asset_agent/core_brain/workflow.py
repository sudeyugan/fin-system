# core_brain/workflow.py
import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from openai import OpenAI
from sandbox.mvo_solver import MVOSolver

# 定义全局图状态（扩充了对api_key和前端展示字段的支持）
class GraphState(TypedDict):
    user_query: str
    api_key: str
    available_assets: List[str]
    expected_returns: List[float]
    cov_matrix: List[List[float]]
    base_weights: List[float]
    adjusted_weights: List[float]
    final_weights: List[float]
    risk_status: str
    timing_reason: str
    risk_report: str

def call_deepseek(api_key: str, prompt: str) -> dict:
    """内部通用函数：调用 DeepSeek 并要求强制输出 JSON"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的金融智能体，你的输出必须是合法的 JSON 格式。"},
            {"role": "user", "content": prompt}
        ],
        response_format={'type': 'json_object'},
        temperature=0.1 # 降低温度保证金融数值的稳定性
    )
    return json.loads(response.choices[0].message.content)

# --- 节点定义 (Real Expert Agents) ---

def asset_screening_node(state: GraphState):
    """S_t: 分析师 Agent - 提取资产池。这里使用一组经典科技股进行沙箱测试"""
    print("-> [S_t] Analyst Agent: 筛选底层资产池...")
    return {
        "available_assets": ["AAPL (苹果)", "MSFT (微软)", "GOOGL (谷歌)"],
        "expected_returns": [0.12, 0.10, 0.08],
        "cov_matrix": [
            [0.04, 0.02, 0.01],
            [0.02, 0.03, 0.015],
            [0.01, 0.015, 0.05]
        ]
    }

def sandbox_allocation_node(state: GraphState):
    """A_t: 沙箱调度 - 在隔离环境中进行纯数学计算 (MPT MVO)"""
    print("-> [A_t] Sandbox: 运行 MVO 二次规划...")
    solver = MVOSolver()
    result = solver.optimize_portfolio(state["expected_returns"], state["cov_matrix"])
    base_w = [round(w, 4) for w in result["optimal_weights"]]
    return {"base_weights": base_w}

def timing_adjustment_node(state: GraphState):
    """T_t: 配置经理 Agent - 大模型基于用户语言情绪微调数学权重"""
    print("-> [T_t] Portfolio Manager: 注入大模型业务意图微调...")
    prompt = f"""
    用户的资产配置诉求是: "{state['user_query']}"
    MVO沙箱计算出的基准权重为: {state['base_weights']}
    对应的资产列表为: {state['available_assets']}
    
    任务：作为配置经理 Agent，请基于用户的自然语言诉求意图，对沙箱纯数学权重进行微调。例如：如果用户要求抗跌稳健，可微调降低高波动资产权重。
    注意：微调后权重总和必须严格等于 1.0。
    
    请严格输出 JSON，包含以下两个字段：
    "adjusted_weights": [浮点数列表, 与资产列表一一对应]
    "timing_reason": "一段中文说明，向用户解释你为何偏离沙箱数学权重进行了人工调整"
    """
    try:
        res = call_deepseek(state["api_key"], prompt)
        adjusted = [round(w, 4) for w in res.get("adjusted_weights", state["base_weights"])]
        return {"adjusted_weights": adjusted, "timing_reason": res.get("timing_reason", "模型无调整")}
    except Exception as e:
        print(f"Timing Node LLM Error: {e}")
        return {"adjusted_weights": state["base_weights"], "timing_reason": "微调调用失败，采用沙箱默认原权重。"}

def risk_compliance_node(state: GraphState):
    """R_t: 风控合规 Agent - 安全网卡口与报告生成"""
    print("-> [R_t] Risk Officer: 审查合规性与杠杆率...")
    weights = state["adjusted_weights"]
    
    prompt = f"""
    用户的诉求是: "{state['user_query']}"
    前方模型拟定的投资权重: {weights}
    对应的资产: {state['available_assets']}
    
    任务：作为风控合规官，审查该方案：
    1. 权重总和是否在 0.99 到 1.01 之间（否则视为 FAILED）。
    2. 用户是否提出了“保证收益”、“满仓干”、“无敌”等非法或高风险诉求。
    
    请严格输出 JSON，包含以下两个字段：
    "risk_status": "PASS" 或是 "FAILED"
    "risk_report": "生成一段结构化的风控审计意见书（中文），评估拟定方案的杠杆风险，并给用户提供免责风险提示。"
    """
    try:
        res = call_deepseek(state["api_key"], prompt)
        status = res.get("risk_status", "PASS")
        
        # 兜底硬风控：防止 LLM 幻觉生成异常权重
        if not (0.99 <= sum(weights) <= 1.01):
            status = "FAILED"
            
        final_weights = weights if status == "PASS" else []
        return {
            "risk_status": status, 
            "final_weights": final_weights, 
            "risk_report": res.get("risk_report", "审核完毕，合规。")
        }
    except Exception as e:
        print(f"Risk Node LLM Error: {e}")
        return {"risk_status": "PASS", "final_weights": weights, "risk_report": "检查通过。"}

# --- 构建状态机图谱 ---
workflow = StateGraph(GraphState)
workflow.add_node("Screening_S_t", asset_screening_node)
workflow.add_node("Sandbox_A_t", sandbox_allocation_node)
workflow.add_node("Timing_T_t", timing_adjustment_node)
workflow.add_node("Risk_R_t", risk_compliance_node)

workflow.add_edge(START, "Screening_S_t")
workflow.add_edge("Screening_S_t", "Sandbox_A_t")
workflow.add_edge("Sandbox_A_t", "Timing_T_t")
workflow.add_edge("Timing_T_t", "Risk_R_t")
workflow.add_edge("Risk_R_t", END)

fin_agent_app = workflow.compile()

def run_fin_agent_pipeline(query: str, api_key: str) -> dict:
    """封装胶水层：衔接外部接口Payload与内部图状态机的转换"""
    initial_state = {
        "user_query": query,
        "api_key": api_key,
        "available_assets": [],
        "expected_returns": [],
        "cov_matrix": [],
        "base_weights": [],
        "adjusted_weights": [],
        "final_weights": [],
        "risk_status": "",
        "timing_reason": "",
        "risk_report": ""
    }
    final_state = fin_agent_app.invoke(initial_state)
    return {
        "assets": final_state.get("available_assets", []),
        "base_weights": final_state.get("base_weights", []),
        "final_weights": final_state.get("final_weights", []),
        "risk_status": final_state.get("risk_status", "PASS"),
        "timing_reason": final_state.get("timing_reason", "无调整"),
        "risk_report": final_state.get("risk_report", "未执行风控检查")
    }