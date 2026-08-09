---
title: easy-langent 学习笔记
date: 2026-08-07
tags:
  - 学习
  - Agent
  - LangChain
---

# easy-langent 学习笔记

> [!note] 说明
> DataWhale easy-langent 项目学习笔记,入口见 [[agent目前学习沉淀]]。

## 项目简介

- 仓库:https://github.com/datawhalechina/easy-langent
- 在线阅读:https://datawhalechina.github.io/easy-langent/
- 目标:打破"理论学习"与"实战开发"壁垒,用 LangChain、LangGraph 框架做智能体开发
- 知识主线:**LangChain(组件)→ 应用设计+RAG → LangGraph(图工作流)→ 多智能体 → 综合实战**

---

## 第一章 LangChain与LangGraph框架认知

**一句话**:LangChain 是"乐高积木"(快速搭简单应用),LangGraph 是"建筑设计图"(管控复杂流程),两者互补融合。

### 为什么要学(传统开发的三大痛点)
- 重复造轮子:每次都要自己写模型调用、对话管理
- 流程管控复杂:多步骤任务容易出错
- 状态维护困难:中间数据不好保存传递

### 核心认知
- **LangChain = 基础设施工具箱**:简单~中等复杂度应用。1.0 后拆三层:
  - `langchain_core`(核心抽象 Runnable/BaseParser)、`langchain`(高级组件 Chain/Memory)、`langchain_openai`(模型适配)
- **LangGraph = 架构设计框架**:多步骤、状态管理、多智能体协作、人机交互
- **关系**:从属(LangGraph 依赖 LangChain 生态)+ 互补(简单任务用 Chain,复杂任务用 Graph,实际常混用)
- 类比:积木拼小房子 vs 积木+设计图建高楼

### 环境搭建(面试常问)
- Python 3.10+,虚拟环境(venv/conda/uv 三选一)
- 安装:`pip install langchain langgraph langchain-openai python-dotenv`(**必须 1.0.0+**,旧版不兼容)
- 密钥:`.env` 存 `API_KEY` / `BASE_URL`,`load_dotenv()` 读取,不要硬编码
- 国内加速:清华 PyPI 镜像

### Hello World 对比(最能体现两个框架差异)
- **LangChain**:`llm = ChatOpenAI(...)` → `llm.invoke(prompt)` → 取 `.content`
- **LangGraph 四步法(全书骨架)**:
  1. `TypedDict` 定义 **State**
  2. 定义**节点函数**(接收 state,返回要更新的字段)
  3. `StateGraph(State)` → `add_node` / `add_edge(START→...→END)` → `compile()`
  4. `app.invoke(初始状态)` 得到最终状态
- 结论:Chain 核心是"组件拼接",Graph 核心是"状态+节点+流程"

---

## 第二章 LangChain核心组件实操

**一句话**:围绕"输入可控制、输出可预期",三大组件 = 模型调用 + 提示词模板 + 输出解析。

### 1. 模型调用(ChatOpenAI 统一接口)
- **LLM vs ChatModel**:LLM 文本→文本(如本地 HuggingFace 模型);ChatModel 消息列表→消息(如 GPT、DeepSeek),多轮对话用 ChatModel
- **消息角色 = 约束层级**(面试亮点):`system` 设定整体行为规则(最高优先级,持续生效)> `user` 当前任务 > `assistant` 对话历史(模型本身不记忆,靠回传 assistant 消息保持上下文)
- 统一接口的价值:换模型(OpenAI→HuggingFace)只需改初始化部分,调用逻辑不变
- 调用:`chat_model.invoke(messages)` → `result.content`

### 2. 提示词模板(PromptTemplate)
- 基础:`PromptTemplate(input_variables=[...], template="...{参数}...")` → `.format()` 填参数,模板复用
- **少样本 FewShotPromptTemplate**:`examples`(示例) + `example_prompt`(示例模板) + `suffix`(最终需求),给模型看示例它就照着格式输出
- 工程化:示例存 JSON 文件(便于维护) + **ExampleSelector 动态筛选**(按难度/长度选示例,控制 token,如自定义 `BaseExampleSelector`)

### 3. 输出解析(OutputParser)
| 解析器 | 特点 | 适用 |
|---|---|---|
| StrOutputParser | AIMessage→str,最稳最简 | 文本总结、作为底座 |
| JsonOutputParser | →dict,配置简单不校验 | 快速 Demo |
| PydanticOutputParser | BaseModel 强类型校验,错了直接报错 | **工程默认主线** |
| 自定义(BaseOutputParser) | 实现 `parse()` + `get_format_instructions()` | 特殊格式需求 |

### LCEL 管道(贯穿全书)
`chain = prompt | llm | parser`,前一个输出自动作为后一个输入,像工厂流水线。

---

## 第三章 LangChain进阶组件:Memory 与 Tool

**一句话**:Memory 让 AI"记住话"(状态),Tool 让 AI"动手做事"(行动),组合起来就是真正的智能体。

### 3.1 记忆(Memory)——解决 LLM 无状态
- 本质两个动作:**存**(每轮 HumanMessage+AIMessage 入库)+ **取**(新对话时取出注入 Prompt)
- 技术骨架:`BaseChatMessageHistory`(接口: messages/add_message)+ `InMemoryChatMessageHistory`(内存实现)+ `RunnableWithMessageHistory`(包装链:调用前注入历史、调用后自动存档)+ **session_id**(不同用户记忆隔离)
- 三种记忆对比(面试必答):
  - **全量记忆**:完整上下文,无丢失;但 token 随轮数线性增长 → 短对话
  - **窗口记忆**:只保留最近 N 轮,早期信息被截断 → 长对话、客服
  - **摘要记忆**:LLM 把历史压成摘要注入,省 token 但丢细节 → 超长对话
- 可手动实现理解原理:历史列表 + 拼接 + 调用 + 追加

### 3.2 工具(Tool)——工具调用 = 思考→行动→反馈
- 三组件:**Tool**(具体工具,有名字和描述,AI 靠描述选工具)、**Toolkit**(工具包)、**Agent**(指挥官,协调 LLM 和工具)
- 自定义工具:`@tool` 装饰器 + 函数;三个要求:**docstring 写清楚**(告诉 AI 干什么)、**参数类型注解**、**返回结果清晰**;`args_schema=` 用 Pydantic 校验参数
- 常用参数:`return_direct`(是否跳过 LLM 直接返回工具结果)、`parse_docstring`
- 内置工具:`FileManagementToolkit`(ReadFile/WriteFile/ListDirectory 等)
- 创建 Agent:`create_agent(model=llm, tools=[...], debug=True)`

### 3.3 组合实践(核心思路:模块化拆分 + 流水线组合)
- 拆:记忆(状态)/工具(行动)/模板(输入)/LLM(思考)
- 组:`RunnableLambda(判断) | prompt | llm`,再包 `RunnableWithMessageHistory`
- 案例:带记忆的计算助手(检测到计算→调 PythonREPLTool→LLM 解释)、带记忆的文件助手(`llm.bind_tools(tools)`)

---

## 第四章 应用级系统设计与RAG

**一句话**:两大能力——链式工作流(把复杂任务拆成流水线)+ RAG(给大模型装实时知识库,解决知识滞后和幻觉)。

### 4.1 链式工作流(Runnable 体系)
- 为什么拆:复杂任务 hold 不住、出错难定位、不能灵活换模型/工具
- 核心组件(面试必背):
  - `RunnableSequence` / `|`:线性顺序执行
  - `RunnableBranch`:条件路由,(判断函数, 目标链) 元组定义分支
  - `RunnableParallel` / `RunnableMap`:并行执行多个子组件
  - `RunnableLambda`:把任意函数包装成 Runnable
  - `RunnablePassthrough`:透传原始输入(保留上下文)
- 多输入多输出:`RunnableMap({"卖点": chain, "人群": passthrough}) | 下一prompt`
- **RouterChain 三件套**:目标链(各场景链)+ 路由选择器(LLM 解析需求输出场景标识)+ 默认链(兜底)
- **错误处理三件套**(工程亮点):
  - `with_retry()`:重试临时错误(网络/超时),指数退避+抖动
  - `try-except`:捕获可预知错误(KeyError 缺变量、解析失败),注意先捕获具体异常
  - `with_fallbacks()`:核心链失败自动切降级链(如 GPT-4→GPT-3.5),保证可用性

### 4.2 RAG 核心
- **痛点**:知识滞后(训练数据截止时间)+ 幻觉(一本正经胡说八道)
- **逻辑三步**:检索(从外部知识库找相关内容)→ 增强(问题+资料拼进 Prompt)→ 生成(基于资料作答)
- **价值**:提升准确性、拓展知识边界、不用重训模型(改知识库即可,省微调成本)
- 适用:企业知识库/最新信息/私有数据问答;不适用:创作、推理、闲聊

### 4.3 RAG 五步流水线(实操主线)
1. **文档加载**:TextLoader / PyPDFLoader / Docx2txtLoader / UnstructuredMarkdownLoader → 统一 `Document` 对象(内容+元数据)
2. **文本分割**:`RecursiveCharacterTextSplitter(chunk_size=200-500, chunk_overlap=50, 分隔符优先级)`;原则:语义相关 + 大小适中;MD 用 MarkdownTextSplitter
3. **向量存储**:嵌入模型(HuggingFaceEmbeddings, 如 Qwen3-Embedding)→ `FAISS.from_documents()` → `save_local()`
4. **检索**:`vector_db.as_retriever()`;**similarity**(精准) vs **MMR**(相关+多样);参数 k(建议3-4)、lambda_mult(0.7-0.8 偏相关)
5. **生成(LCEL 整合)**:`{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`

### 4.4 评估与调优(加分项)
- 指标:相关性(Precision/Recall/F1)、准确性(事实一致性)、响应速度
- 工具:RAGAS 自动化评估(不需要标注答案)
- 调优方向:分割策略(chunk 大小)、向量模型选择(中文用 Qwen3-Embedding 等)、检索参数
- 闭环:定位问题→针对性调优→验证→迭代

---

## 第五章 中期综合实践:设计一个智能体项目

**一句话**:不学新知识,把前四章组件整合成一个"可运行、有实际用途"的智能体应用,完成从"组件使用者"到"应用设计者"的跃迁。

### 功能技术要求(完成参考线)
1. **模型与提示层**:PromptTemplate / 少样本,参数化+标准化
2. **链式工作流**:用 Runnable / `|`,至少 2 个以上步骤
3. **状态与行动能力(核心)**:Memory 或 Tool 二选一,且必须真正被使用(不能形式化)
4. **输出可控**:OutputParser 或提示词约束固定格式(如 JSON)

### 需求拆解方法
明确核心需求(一句话:用户是谁/解决什么问题/要什么结果)→ 拆成 3-6 个可执行子任务 → 每个子任务对应一个 LangChain 组件

### 常见坑
- 把所有逻辑塞进一个大 Prompt(要拆分)
- Memory/Tool 加了但没用(设计时就要明确用途)
- 代码不工程化(无注释、无模块拆分)
- 功能过度堆砌(聚焦一个核心功能做透)

### 交付与自检(面试时能讲出来)
- 代码可一键运行 + README(背景/流程图/组件清单/运行步骤)
- 自检:能否讲清每个链的作用?能否画出系统流程图?去掉 Memory/Tool 能力是否明显下降?能否解释设计取舍?

---

## 第六章 LangGraph基础:有状态工作流

**一句话**:LangGraph = 用"图"建模工作流,原生支持分支/循环/并行,三大组件 = 状态(黑板)、节点(工人)、边(路线)。

### 为什么需要图结构(对比 LangChain)
- 线性链"刚性":步骤固定、难动态调整、状态分散在链里
- LangGraph 优势:显式控制(流程可视化)、可预测、精细调控(下一步去哪、何时循环/终止)
- 对比表:流程灵活性(链式 vs 图)、状态管理(分散 vs 统一状态对象+字段级合并+持久化)、分支循环(需手写 vs 原生支持)

### 三大组件(面试必答)
- **State(共享黑板)**:`TypedDict` 定义;字段原则:最小必要/可更新/清晰命名;**不可变更新**:节点只返回要更新的字段,框架自动合并成新状态传给下一节点
- **Node(功能小工人)**:纯函数"读状态→执行→返回更新";三类:LLM 节点/工具节点/数据处理节点;**禁止节点间直接调用**,只能通过状态交互
- **Edge(路径导航员)**:
  - 固定边 `add_edge`:线性,必定跳转
  - 条件边 `add_conditional_edges(起点, 路由函数, {分支名: 目标节点})`:动态决策
  - 循环边:条件边特殊形式(回环到前序节点),**必须设终止条件防死循环**

### 运行机制(超步骤 Super-step)
- 每轮"超步骤"三件事:激活节点(收到状态更新的节点变活跃)→ 并行执行 → 传递消息(状态片段)给下一节点
- 顺序执行:单后续+强依赖;并行执行:多后续+无依赖(如同时生成摘要和关键词,扇出后汇合)

### 综合实操三案例(线性→分支→循环)
1. **线性**:文本去重→摘要→敏感词校验→输出(固定边)
2. **分支**:质量校验不合格 → 回退重生成,`rewrite_count` 限制重试次数,超限强制输出
3. **循环+人机交互**:AI 优化→用户反馈(确认/修改/退出)→路由回优化节点或结束
- 配套:图可视化 `draw_mermaid_png()`、`MemorySaver`+`thread_id` 看状态历史

---

## 第七章 LangGraph进阶:多智能体协作与复杂流程管控

**一句话**:多智能体 = 把一个复杂任务拆给多个"专业智能体"组队干活;三种架构 + 子图/并行/循环 + 人机协作,构建生产级系统。

### 7.1 为什么需要多智能体
- 解决单一 LLM **长指令疲劳**(任务越多输出越不稳定)+ 上下文污染;类比:一个人又做饭又洗碗会出错,厨师+洗碗工分工效率高
- 模块化工程思想:每个智能体独立开发/测试/修改,加功能只增节点不动全局

### 7.1.2 三种架构模式(面试必背)
1. **中心化 Supervisor(主管-员工)**:主管接收任务→拆分→分派→汇总;适用任务可明确拆分;注意**轮次上限**(MAX_ROUNDS 兜底)
2. **链式 Sequence**:无主管,固定顺序接力(写→纠错→润色),前一个输出是后一个输入;适用流程固定
3. **去中心化 Peer-to-peer**:平等智能体,基于全局状态自主判断是否干活(看状态更新→待办任务→项目目标);适用流程灵活

### 7.1.3 通信机制
- **全局 State 共享**:公共白板,读写即通信
- **差异化 System Prompt**:明确职责边界 + 通信约定(如主管只输出下一个智能体名)+ 统一输出格式

### 7.2 复杂流程管控
- **子图 Subgraph**:图里嵌图(先编译子图,再当节点加进主图),实现逻辑隔离+复用
- **并行(扇出/扇入)**:一个节点分派多个智能体并行干 → 汇总节点整合;**并发冲突两种解法**:① 各并行节点用独立状态键(推荐);② 同字段用自定义合并函数
- **循环与重试**:审核不达标→回退重生成(条件边回环);**循环次数限制**防死循环

### 7.3 人机协作(Human-in-the-loop)
- **检查点 MemorySaver**:每步状态存档(节点输出+state+下一节点),`thread_id`/session_id 恢复进度,崩溃续跑
- **中断 Interrupts**:`interrupt_before`(关键操作执行前人工授权,如发邮件/转账)、`interrupt_after`(执行后人工确认)
- **动态状态编辑**:`update_state()` 人工修改中间输出;或设置 `current_node` 把工作流**回退到任意节点**重新执行

### 7.4 综合实践:小说创作助手
- 流程:用户输入 → LLM 生成题目/角色/情节 → **用户审核确认(条件分支,不过则重生成)** → 生成大纲章节 → 生成小说
- 对应知识点:节点设计、状态传递、条件边、人机协同

---

## 第八章 综合实践:构建"谁是卧底"游戏智能体

**一句话**:用 LangGraph 把一个多轮博弈游戏做成智能体引擎——6 个模块 = 6 个节点,条件边控制游戏循环,是全书知识的完整串联。

### 游戏规则(简化)
- 4 个智能体(1 卧底 + 3 平民);平民词与卧底词高度相似(奶茶-果汁)
- 流程:发言(描述特征不能说词语)→ 投票 → 淘汰得票最多者;淘汰卧底→平民胜;剩 1 民 1 卧→卧底胜;否则进入下一轮
- 默认"上帝视角"(用户旁观,全智能体自主),可选"玩家视角"(用户参与)

### 6 个核心模块 = 6 个节点
1. **词语生成**:LLM 出题,强制 JSON 输出 + **解析失败兜底**(从备用词库随机选)
2. **角色分配**:随机指定卧底,其余平民
3. **发言生成**:结合历史发言(多轮记忆),**平民真实描述 vs 卧底伪装**,Prompt 约束 10-100 字,排除已淘汰者
4. **投票**:基于当前轮+历史发言找矛盾(卧底常前后矛盾),**多重兜底**:不投自己/不投淘汰者/解析失败随机投
5. **胜负判断**:统计票数淘汰→判断胜负或进入下一轮
6. **结果展示**:输出胜利方、词语、轮次、淘汰顺序

### 图结构
- 固定边串联:生成词语→分配角色→发言→投票→裁决
- **条件边(核心)**:裁决后 `game_status == "running"` → 回"发言"节点循环;否则 → 结果展示 → END

### 工程要点(面试可说)
- LLM 输出一律 JSON + try-except 兜底,保证程序稳定
- `history_speeches` 让智能体拥有跨轮记忆
- 提交规范:project 文件夹建个人目录(驼峰命名),GitHub PR 提交,含核心代码 + Readme,**禁止提交 .env**

---

## 整体知识体系(面试串联话术)

> 我可以先用 **LangChain** 的三大组件(模型调用、提示词模板、输出解析)搭出"输入可控、输出可预期"的基础应用;加上 **Memory 和 Tool** 让应用有记忆、能动手;面对复杂任务用**链式工作流(线性/路由/并行)+ 错误处理**拆解流程,用 **RAG** 让模型基于私有知识库回答;当流程复杂到需要分支、循环、多智能体协作时,升级到 **LangGraph**,用"状态-节点-边"建模,配合子图、并行、人机协作(中断/检查点)实现生产级系统;最后通过"谁是卧底"这类实战把全部知识串成完整的多智能体应用。
