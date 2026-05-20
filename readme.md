# FinAgent OS — 现代智能资产配置系统

FinAgent OS 是一个基于 **控制与计算分离（Separation of Control and Computation）** 架构设计、面向现代财富管理的智能资产配置系统。系统利用大语言模型（LLM）的自然语言意图理解能力，融合高精度确定性数学沙箱与动态多源数据流（DataOps），通过 LangGraph 状态机工作流，为用户提供端到端的虚拟画像构建、实时资产筛选、马科维茨均值-方差优化（MVO）、配置意图微调以及合规风控审计功能。

---

## � 快速开始与使用方法

### 1. 环境准备
确保已安装 Python 3.10+，并在项目根目录安装相关依赖：
```bash
# 创建并激活虚拟环境（推荐）
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动核心 API 服务
启动基于 FastAPI 的编排工作流后端，向前端提供由 LangGraph 驱动的配置接口：
```bash
# 进入核心业务模块
cd fin_asset_agent

# 启动服务器（默认运行在 http://127.0.0.1:8000）
python main_api.py
```

### 3. 访问前端交互套件
- 保持后端服务运行，在浏览器中直接双击打开 [fin_asset_agent/index.html](fin_asset_agent/index.html) 或将其拖入浏览器中。
- 在前端操作面板的配置区域输入你的 **DeepSeek API Key**。
- 在对话框输入你的大白话个人资产规划需求（*例如："我有50万，目前单身，风险承受能力还可以，帮我配置一些A股蓝筹以及部分美股标的"*）。
- 系统将在界面实时展示语言模型推理、纯数学沙箱重算、以及人机意图微调对齐的全维审计流演进过程。

---

## �🛠️ 项目当前开发进展综述

根据当前代码库的审查，系统已完成了**核心链路的闭环搭建（端到端 Pipeline 已完全跑通）**，但在具体的文件组织架构上，目前采用的是**高内聚的紧凑型实现**（许多蓝图中的细分 Agent 逻辑暂时内聚于 `core_brain/workflow.py` 状态机节点中，数据抓取也完全收拢在 `data_ops/market_data.py`）。

### ✅ 已完成的核心功能与架构特性
1. **控制与计算的彻底分离**：
   * **控制（LLM/Graph）**：由 DeepSeek-V4 提供大模型服务，配合 LangGraph 构建动态状态机，实现意图路由、画像创设、情绪微调、合规审查。
   * **计算（Sandbox）**：由 NumPy、SciPy 基于马科维茨现代资产配置理论（MPT）在隔离的沙箱环境中执行确定性的二次规划（SLSQP）求解，杜绝大模型数值幻觉。
2. **多源 DataOps 量化管线与混合缓存**：
   * 实现了**境内 AkShare 接口（高速拉取 A 股前复权日线）**与**境外 yfinance 接口（拉取全球资产）**的自动分流与数据合并。
   * 建立了本地日线快照缓存机制（`data_ops/cache/`），同日内重复请求直接命中快照，跳过网络 IO，保障系统健壮性。
   * 内置极限边界逃生通道（Fallback），当网络彻底阻断或 API 限流时，触发自动确定性模拟数据降级，保证状态机完整走完。
3. **LangGraph 工业级状态机设计**：
   * 实现了包含 `Profile_P_t` ➡️ `Screening_S_t` ➡️ `Sandbox_A_t` ➡️ `Timing_T_t` ➡️ `Risk_R_t` 完整的有向无环图（DAG）状态机。
4. **前后端解耦与全功能 Web 交互**：
   * **后端**：使用 FastAPI 搭建高并发 API 服务，一键拉起跨域（CORS）中枢，无缝编排整个 LangGraph Pipeline。
   * **前端**：构建了冷冽聚焦、纯白微悬浮面板风格的现代大厂风 Web 界面（支持 ECharts 极简双曲线覆盖图表演进对照、系统实时日志流追踪、API 密钥动态注入等）。

### ❌ 未完成或待改进的进阶工作
1. **Agent 代码解耦（待重构）**：目前的 `analyst`、`portfolio_mgr` 和 `risk_officer` 节点代码均直接写在 `core_brain/workflow.py` 内，未来需要按最初的蓝图规划拆分到 `core_brain/agents/` 子目录下进行模块化工程管理。
2. **非结构化数据层（RAG 引擎未填充）**：`rag_engine.py` 和 `text_extract.py` 尚未进行实质填充。当前资产筛选属于确定性硬编码，未真正接入券商财务报表、宏观研报的非结构化知识库切片。
3. **回测引擎与状态长期记忆层（未实现）**：`backtest_engine.py`、`db_models.py` 与 `state_manager.py` 均为空文件或尚未在当前主干工作流中承接逻辑。缺乏多轮对话的 Context 选择性切片记忆，以及基于 Backtrader 的历史调仓策略回测验证。

---

## 📂 代码库文件功能详述（对照蓝图规范）

以下是 FinAgent OS 系统的代码文件功能全景对照表，包含**已填充核心逻辑的代码**以及**蓝图规范中待后续填充/扩展的代码**：

### 1. 根目录与入口编排层
* **`main_api.py`【已填充核心逻辑】**
  * **功能**：作为编排层 API 服务入口，基于 FastAPI 构建。支持跨域请求（CORS），向 Web 前端暴露 `/api/allocate` 统一标准的 POST 接口。
  * **逻辑**：接收前端传入的 `query` 与 `api_key`，将其转化为图状态机的初始 Payload，启动 LangGraph 引擎，并将最终的风控状态、微调理由、优化权重和虚拟档案以统一标准的 JSON 格式吐给前端。
* **`index.html`【已填充核心逻辑】**
  * **功能**：FinAgent OS 的全功能现代化前端工作台。
  * **视觉与交互**：基于 Tailwind CSS 实现纯白微悬浮拟物面板。集成 ECharts 图表，动态绘制**纯数学基准线（Base）**与**LLM 意图调整线（Final）**的多维重叠演进对照；提供高可见度的实时系统时钟、系统运行日志流。
* **`app.py`【规范预留 — 暂未填充】**
  * **功能**：最初蓝图规划中的 Streamlit 交互层入口。当前系统已升级为体验更佳的 FastAPI + 精美纯前端 HTML/JS 方案，此文件可作为后续多模态看板或纯 Python 演示环境的备用入口。
* **`config.py`【规范预留 — 暂未填充】**
  * **功能**：全局静态参数配置文件。未来可用于统筹管理数据库连接字符串（SQLite/Postgres）、LLM 基础 URL、默认温度参数、多轮记忆滑动窗口大小等硬编码常量。
* **`requirements.txt`【已填充】**
  * **功能**：声明项目依赖的底层基础库（包括 `fastapi`, `uvicorn`, `langgraph`, `numpy`, `scipy`, `yfinance`, `pandas`, `akshare`, `pydantic` 等）。

### 2. `core_brain/` — 编排与控制层
* **`core_brain/router.py`【已填充核心逻辑】**
  * **功能**：系统意图路由中枢（Core Router）。
  * **逻辑**：利用 Pydantic 约束输入输出，调用 DeepSeek 模型通过強约束的 JSON Object 模式，将用户的自然语言提问精准路由至四大核心意图：`QA_RAG`（问答检索）、`ASSET_ALLOCATION`（资产配置）、`PORTFOLIO_REVIEW`（组合审查）或 `CHIT_CHAT`（闲聊），并抽取核心实体（时间、标的）。
* **`core_brain/workflow.py`【已填充核心逻辑】**
  * **功能**：LangGraph 状态机定义与专家 Agent 胶水合并层。
  * **内部节点逻辑**：
    * `profile_generation_node` (`P_t`)：基于用户诉求，调用 LLM 扮演顾问，动态量身创设合理的虚拟资产档案与资产存量存量结构。
    * `asset_screening_node` (`S_t`)：确定核心测试标的（贵州茅台、工商银行、比亚迪），并驱动 DataOps 模块去动态抓取历史时间序列指标。
    * `sandbox_allocation_node` (`A_t`)：调度 `MVOSolver` 进行物理沙箱隔离计算。
    * `timing_adjustment_node` (`T_t`)：注入 LLM 业务意图，基于诉求对纯数学权重执行微调，并强制限定调整后权重总和为 1.0。
    * `risk_compliance_node` (`R_t`)：风控合规官节点。审查方案是否满足严格的风控限额（总和 0.99~1.01 约束，非法/高风险词汇过滤），并开具结构化的中文风控审计意见书。
* **`core_brain/agents/`【规范预留 — 暂未填充/需解耦】**
  * **`analyst.py` / `portfolio_mgr.py` / `risk_officer.py`**
  * **后续方向**：需将目前堆叠在 `workflow.py` 中的节点函数剥离并重构入此目录下。每一个独立的文件应当承载更复杂的 Prompt 模版、长文本审计链路及自主 Tool Calling（工具调用）逻辑。

### 3. `data_ops/` — 数据与知识检索层
* **`data_ops/market_data.py`【已填充核心逻辑】**
  * **功能**：DataOps 混合量化计算引擎。
  * **高可用逻辑**：提供 A/B 双路方案。方案 A 优先命中本地生成的今日 CSV 快照；方案 B 触发多源网络拉取，其中自动识别 A 股标的调用国内 `AkShare` 日线接口，美股等全球资产调用 `yfinance` 全球接口。在合并矩阵时利用 Pandas 执行 `ffill().bfill()` 完美对齐中美交易节假日错配，最终输出 252 交易日等比缩放的**年化预期收益率向量**和**年化协方差矩阵**。
* **`data_ops/rag_engine.py`【规范预留 — 暂未填充】**
  * **功能**：非结构化专家知识库检索增强引擎。后续将规划引入 Chroma/Milvus 等向量数据库，利用父子块切片（Parent-Child Retriever）技术实现精准的金融知识和逻辑检索。
* **`data_ops/text_extract.py`【规范预留 — 暂未填充】**
  * **功能**：金融研报与复杂上市公司财报（PDF/Excel）的文本解析工具，用于清洗、提取并源源不断地为 RAG 知识库喂入底料。

### 4. `sandbox/` — 确定性计算沙箱层
* **`sandbox/mvo_solver.py`【已填充核心逻辑】**
  * **功能**：马科维茨均值-方差优化（Mean-Variance Optimization, MVO）纯数学求解器。
  * **计算核心**：基于国债无风险利率基准（默认 2%），通过对目标函数（组合方差 $Weights^T \times Cov \times Weights$）进行基于 SLSQP 方法的二次规划非线性求解。同时施加多头做多约束（Bounds $0.0 \sim 1.0$）与权重归一化硬约束（$\sum w_i = 1.0$），保证最优化数值绝对精确稳定。
* **`sandbox/backtest_engine.py`【规范预留 — 暂未填充】**
  * **功能**：Backtrader 历史策略回测引擎集成。计划后续承接 `workflow.py` 吐出的微调权重，在历史长周期内模拟真实调仓，计算夏普比率、最大回撤等回测指标。
* **`sandbox/visualizer.py`【规范预留 — 暂未填充】**
  * **功能**：离线情况下的 Python 绘图工具。由于目前系统已采用基于 FastAPI + 前端 ECharts 动态异步渲染的可视化路线，此模块未来可用于离线生成 PDF 审计报告时，通过 Matplotlib/Plotly 静态绘制有效前沿面（Efficient Frontier）。

### 5. `memory/` — 状态与长期记忆层
* **`memory/db_models.py`【规范预留 — 暂未填充】**
  * **功能**：结构化长期数据库模型定义，通常基于 SQLAlchemy 映射至 SQLite 或 PostgreSQL。后续用于持久化存储用户的历史持仓快照、历史风险评估得分及实盘调仓指令。
* **`memory/state_manager.py`【规范预留 — 暂未填充】**
  * **功能**：多轮会话 Context 上下文控制中枢。支持选择性切片记忆，防止长文本多轮对话导致的 Token 爆量或记忆丢失，实现跨会话的记忆追踪。

---

## 🚀 后续功能演进与优化指南

为使系统从目前的“工程原型（MVP）”真正迈向“工业级生产环境”，建议从以下方向依次迭代：

1. **解耦智能体模块**：将 `workflow.py` 中庞大的 LLM Prompt 分离至 `core_brain/agents/` 中。为 `risk_officer.py` 编写更严苛的合规条款，为 `portfolio_mgr.py` 引入更丰富的宏观经济周期情绪因子。
2. **填充非结构化数据链（RAG）**：启动 `text_extract.py` 与 `rag_engine.py`，允许系统通过上传 PDF 研报动态改变资产池名单，取代目前在 `asset_screening_node` 中固定的硬编码 A 股标的。
3. **闭环历史回测**：激活 `backtest_engine.py`，实现 “用户输入 ➡️ 沙箱优化 ➡️ LLM 微调 ➡️ **历史回测校验（夏普、回撤评估）** ➡️ 风控审计 PASS” 的全自动化双向验证科学管线。