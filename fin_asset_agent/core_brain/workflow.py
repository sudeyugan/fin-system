from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from sandbox.mvo_solver import MVOSolver

# 定义全局图状态
class GraphState(TypedDict):
    user_query: str
    available_assets: List[str]
    expected_returns: List[float]
    cov_matrix: List[List[float]]
    base_weights: List[float]
    adjusted_weights: List[float]
    final_weights: List[float]
    risk_status: str

# --- 节点定义 (模拟 Expert Agents) ---

def asset_screening_node(state: GraphState):
    """S_t: 分析师 Agent (模拟) - 提取资产池与预期数据"""
    print("-> [S_t] Analyst Agent: 筛选资产池...")
    return {
        "available_assets": ["AAPL", "MSFT", "GOOGL"],
        "expected_returns": [0.12, 0.10, 0.08],
        "cov_matrix": [
            [0.04, 0.02, 0.01],
            [0.02, 0.03, 0.015],
            [0.01, 0.015, 0.05]
        ]
    }

def sandbox_allocation_node(state: GraphState):
    """A_t: 沙箱调度 - 纯数学计算"""
    print("-> [A_t] Sandbox: 运行 MVO 二次规划...")
    solver = MVOSolver()
    result = solver.optimize_portfolio(state["expected_returns"], state["cov_matrix"])
    return {"base_weights": result["optimal_weights"]}

def timing_adjustment_node(state: GraphState):
    """T_t: 配置经理 Agent (模拟) - 基于情绪微调权重"""
    print("-> [T_t] Portfolio Manager: 注入市场情绪微调...")
    base = state["base_weights"]
    # 模拟微调：略微增加第一个资产的权重
    adjusted = [base[0] + 0.05, base[1] - 0.025, base[2] - 0.025]
    return {"adjusted_weights": adjusted}

def risk_compliance_node(state: GraphState):
    """R_t: 风控合规 Agent (模拟) - 检查并输出最终结果"""
    print("-> [R_t] Risk Officer: 审查合规性与杠杆率...")
    # 简单的风控逻辑验证
    weights = state["adjusted_weights"]
    if sum(weights) > 1.01 or sum(weights) < 0.99:
        return {"risk_status": "FAILED: 权重不合法", "final_weights": []}
    return {"risk_status": "PASS", "final_weights": weights}

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