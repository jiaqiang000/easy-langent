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
> 每章末尾附「核心代码模式」——去掉了密钥校验、debug 打印、异常兜底等边缘逻辑,保留最能体现本章知识的骨架代码;看到代码即回忆知识点。

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

### 核心代码模式:两个框架的 Hello World 对比
先看 LangChain——一句话调模型,没有流程概念:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model="deepseek-v4-flash", temperature=0.3)
response = llm.invoke("请写一段50字左右的AI学习建议")   # 传入提示词,直接返回结果
print(response.content)                                # 结果是 AIMessage,取 .content
```

再看 LangGraph——同样的任务(生成→精简),多了"状态+节点+边"的完整流程(全书写工作流的四步法骨架):

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ① State:工作流的"共享数据容器",存中间结果
class WorkflowState(TypedDict, total=False):
    user_role: str
    original_advice: str
    simplified_advice: str

# ② 节点:每个节点是一个函数,接收 state,返回要更新的字段
def generate_advice(state: WorkflowState):
    result = llm.invoke(f"给{state['user_role']}写一段50字左右的AI学习建议")
    return {"original_advice": result.content}

def simplify_advice(state: WorkflowState):
    result = llm.invoke(f"把下面的建议精简到30字以内:{state['original_advice']}")
    return {"simplified_advice": result.content}

# ③ 构建图:add_node 注册节点,add_edge 定义执行顺序
workflow = StateGraph(WorkflowState)
workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", END)
app = workflow.compile()                                # ④ 编译成可执行实例

result = app.invoke({"user_role": "高校学生"})           # ⑤ 执行,返回完整状态(含所有中间结果)
print(result["simplified_advice"])
```

> 对比看:Chain 的核心是"组件拼接",Graph 的核心是"状态+节点+流程"——这就是第一章最核心的结论。

---

## 第二章 LangChain核心组件实操

**一句话**:围绕"输入可控制、输出可预期",三大组件 = 模型调用 + 提示词模板 + 输出解析。

### 1. 模型调用(ChatOpenAI 统一接口)
- **LLM vs ChatModel**:LLM 文本→文本(如本地 HuggingFace 模型);ChatModel 消息列表→消息(如 GPT、DeepSeek),多轮对话用 ChatModel
- **消息角色 = 约束层级**(面试亮点):`system` 设定整体行为规则(最高优先级,持续生效)> `user` 当前任务 > `assistant` 对话历史(模型本身不记忆,靠回传 assistant 消息保持上下文)
- 统一接口的价值:换模型(OpenAI→HuggingFace)只需改初始化部分,调用逻辑不变
- 调用:`chat_model.invoke(messages)` → `result.content`

**核心代码模式**:多轮对话 = 维护消息列表,每轮把 user 和 assistant 消息都追加进历史再调用:

```python
history = [{"role": "system", "content": "你是一个耐心的AI学习助手"}]   # system:全局行为规则

history.append({"role": "user", "content": "请用3句话解释什么是LangChain?"})
result = chat_model.invoke(history)                        # 每次调用都要带上完整历史
history.append({"role": "assistant", "content": result.content})   # 回传assistant消息=模型的"记忆"

history.append({"role": "user", "content": "它的核心组件有哪些?"})  # 模型因此能理解"它"指什么
result = chat_model.invoke(history)
```

### 2. 提示词模板(PromptTemplate)
- 基础:`PromptTemplate(input_variables=[...], template="...{参数}...")` → `.format()` 填参数,模板复用
- **少样本 FewShotPromptTemplate**:`examples`(示例) + `example_prompt`(示例模板) + `suffix`(最终需求),给模型看示例它就照着格式输出
- 工程化:示例存 JSON 文件(便于维护) + **ExampleSelector 动态筛选**(按难度/长度选示例,控制 token,如自定义 `BaseExampleSelector`)

**核心代码模式**:

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 基础模板:固定文本与动态参数分离
prompt_template = PromptTemplate(
    input_variables=["user_role", "subject"],          # 声明动态参数
    template="请给{user_role}写一段50字左右的{subject}学习建议。"
)
formatted = prompt_template.format(user_role="高校学生", subject="LangChain")  # 填参数→完整提示词

# 少样本模板:给模型看示例,让它"照着格式写"
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                                 # 参考示例列表
    example_prompt=example_prompt,                     # 示例的解析模板
    suffix="学科：{new_subject}\n学习方法：",           # 示例之后才是真正的用户需求
    input_variables=["new_subject"]
)
```

### 3. 输出解析(OutputParser)
| 解析器 | 特点 | 适用 |
|---|---|---|
| StrOutputParser | AIMessage→str,最稳最简 | 文本总结、作为底座 |
| JsonOutputParser | →dict,配置简单不校验 | 快速 Demo |
| PydanticOutputParser | BaseModel 强类型校验,错了直接报错 | **工程默认主线** |
| 自定义(BaseOutputParser) | 实现 `parse()` + `get_format_instructions()` | 特殊格式需求 |

**核心代码模式**:三个解析器都接在 LCEL 管道末尾,用法一致(`prompt | llm | parser`):

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

chain = prompt | llm | StrOutputParser()        # ① 输出统一转 str

chain = prompt | llm | JsonOutputParser()       # ② 自动引导模型输出 JSON → Python dict

from pydantic import BaseModel, Field
class ToolInfo(BaseModel):                      # ③ 用 Pydantic 定义目标结构
    tool_name: str = Field(description="工具名称")
    difficulty: str = Field(description="难度，仅可选：简单/中等/复杂")

chain = prompt | llm | PydanticOutputParser(pydantic_object=ToolInfo)
result = chain.invoke(...)                       # 输出直接是 ToolInfo 对象,可 .model_dump()
```

> 衔接:三个组件的共同载体是 **LCEL 管道 `|`**——前一个组件的输出自动成为后一个组件的输入,这条管道从本章起贯穿全书。

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

**核心代码模式**:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 提示词里留一个"历史消息占位符"——记忆注入点
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手"),
    MessagesPlaceholder(variable_name="chat_history"),   # 运行时把历史消息插到这里
    ("human", "{user_input}")
])
base_chain = prompt | llm

store = {}   # session_id → 历史记录(生产环境换成 Redis/数据库)
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# RunnableWithMessageHistory:记忆"调度员",调用前自动取历史、调用后自动存档
chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_history,
    input_messages_key="user_input",       # 输入中用户问题的键名
    history_messages_key="chat_history"    # 与占位符对应
)
chain.invoke({"user_input": "我叫小明，喜欢编程"},
             config={"configurable": {"session_id": "user_001"}})   # session_id 隔离不同用户
```

**三种记忆的区别只藏在 `get_session_history` 这一步**(面试时说这句很加分):

```python
WINDOW_SIZE = 2   # 窗口记忆:只保留最近 N 轮
def get_window_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:          # 每轮=用户+助手2条消息
        history.messages = history.messages[-2 * WINDOW_SIZE:]   # 截断,丢掉早期消息
    return history
# 摘要记忆则是在这一步之前先用 summary_chain 把历史压成一段摘要再注入
```

### 3.2 工具(Tool)——工具调用 = 思考→行动→反馈
- 三组件:**Tool**(具体工具,有名字和描述,AI 靠描述选工具)、**Toolkit**(工具包)、**Agent**(指挥官,协调 LLM 和工具)
- 自定义工具:`@tool` 装饰器 + 函数;三个要求:**docstring 写清楚**(告诉 AI 干什么)、**参数类型注解**、**返回结果清晰**;`args_schema=` 用 Pydantic 校验参数
- 常用参数:`return_direct`(是否跳过 LLM 直接返回工具结果)、`parse_docstring`
- 内置工具:`FileManagementToolkit`(ReadFile/WriteFile/ListDirectory 等)
- 创建 Agent:`create_agent(model=llm, tools=[...], debug=True)`

**核心代码模式**:

```python
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def weather_query(city: str) -> str:
    """查询指定城市天气"""        # docstring = 给 AI 看的说明书,它靠这个决定何时调用
    return {"北京": "晴，-2~8℃", "上海": "多云，5~12℃"}.get(city, "暂无数据")

agent = create_agent(model=llm, tools=[weather_query])   # Agent 负责"思考→行动→反馈"循环
response = agent.invoke({"messages": [{"role": "user", "content": "北京今天的天气怎么样？"}]})
print(response["messages"][-1].content)                  # 工具结果经 LLM 整理后的最终回答
```

### 3.3 组合实践(核心思路:模块化拆分 + 流水线组合)
- 拆:记忆(状态)/工具(行动)/模板(输入)/LLM(思考)
- 组:`RunnableLambda(判断) | prompt | llm`,再包 `RunnableWithMessageHistory`
- 案例:带记忆的计算助手(检测到计算→调 PythonREPLTool→LLM 解释)、带记忆的文件助手(`llm.bind_tools(tools)`)

**核心代码模式**:LCEL 流水线上加任意函数 + 工具绑定,新功能只加模块不改整条链:

```python
from langchain_core.runnables import RunnableLambda

chain = (RunnableLambda(judge_and_calc)   # 自定义判断函数包装成 Runnable,插进管道
         | prompt
         | llm)

agent = prompt | llm.bind_tools(tools)    # 方式2:绑定工具,LLM 自主决定何时调用
```

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

**核心代码模式**:

```python
# ① 线性链:`|` 串联,组件间用 RunnableLambda 转换数据格式
overall_chain = (sell_point_prompt | llm
                 | RunnableLambda(lambda msg: {"sell_points": msg.content})
                 | marketing_prompt | llm)
result = overall_chain.invoke({"product_intro": product_intro})

# ② 多输入: RunnableMap 并行产出 + RunnablePassthrough 透传原始输入
overall_chain = (
    RunnableMap({
        "sell_points": sell_point_prompt | llm | (lambda x: x.content),
        "target_audience": RunnablePassthrough(),
    })
    | marketing_prompt | llm
)

# ③ 路由链:LLM 解析出场景标识 → RunnableBranch 按标识分发,默认链兜底
router_chain = router_prompt | llm | StrOutputParser()     # 输出 "order"/"refund"/"warranty"
full_router_chain = RunnableBranch(
    (lambda x: x["scene"] == "order", order_chain),
    (lambda x: x["scene"] == "refund", refund_chain),
    (lambda x: x["scene"] == "warranty", warranty_chain),
    default_chain
)

# ④ 错误处理:重试临时错误 / 失败降级到备用链
retry_chain = base_chain.with_retry(stop_after_attempt=3,
                                    retry_if_exception_type=(ConnectionError, TimeoutError))
chain_with_fallback = core_chain.with_fallbacks(fallbacks=[fallback_chain],
                                                exceptions_to_handle=(ConnectionError, TimeoutError))
```

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

**核心代码模式**:五步全流程(面试时按这五步讲 RAG 不会乱):

```python
# ① 加载 → ② 分割
loader = TextLoader("knowledge_base/test.txt", encoding="utf-8")
docs = loader.load()                                             # → Document 列表
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)                      # 语义完整、大小适中的片段

# ③ 嵌入 + 向量存储(给文本做"数字指纹")
embeddings = HuggingFaceEmbeddings(model_name="./models/Qwen/Qwen3-Embedding-0___6B")
vector_db = FAISS.from_documents(split_docs, embeddings)
vector_db.save_local("./faiss_db")                               # 持久化,下次直接 load_local

# ④ 检索器:similarity(精准) 或 mmr(相关+多样)
retriever = vector_db.as_retriever(search_type="mmr",
                                   search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7})

# ⑤ 检索-生成(LCEL 整合):先并行取"检索片段+用户问题",再走 prompt→llm→parser
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("RAG系统的核心价值是什么？")
```

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

### 核心代码模式:一个达标项目的骨架(四条要求 → 四段代码)
本章没有新组件,重点是把前四章的东西"组装"起来——对照下面骨架,就能讲清项目是怎么满足 5.2 节四条要求的:

```python
# ① 提示层:ChatPromptTemplate 参数化 + 少样本引导(对应要求1)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是课程资料问答助手，结合历史对话回答"),
    MessagesPlaceholder(variable_name="chat_history"),   # 记忆注入点
    ("human", "{input}")
])

# ② 链式工作流:至少两步,用 | 串联(对应要求2)
chain = RunnableLambda(judge_and_calc) | prompt | llm

# ③ 状态能力:包上记忆(对应要求3,选 Memory 或 Tool 至少一个)
app = RunnableWithMessageHistory(runnable=chain, get_session_history=get_history,
                                 input_messages_key="input", history_messages_key="chat_history")

# ④ 输出可控:接解析器,保证输出可被程序使用(对应要求4)
chain = prompt | llm | JsonOutputParser()
```

> 记住自检三问:每个链在做什么?去掉 Memory/Tool 能力是否下降?能否画出流程图?

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

### 核心代码模式:四大场景
**① 线性流程**(State→节点→图→编译→执行,标准五连):

```python
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END

class TaskState(TypedDict):
    user_query: str                        # 必填字段
    tool_result: NotRequired[str]          # NotRequired:可选字段,存中间结果
    final_answer: NotRequired[str]

def parse_query(state: TaskState):         # 节点:读状态 → 返回"要更新的字段"(不用返回全部)
    return {"tool_result": f"已解析问题:{state['user_query']}", "progress": 30}

def generate_answer(state: TaskState):
    return {"final_answer": f"基于工具结果 -> {state['tool_result']}"}

builder = StateGraph(TaskState)
builder.add_node("parse", parse_query)
builder.add_node("generate", generate_answer)
builder.add_edge(START, "parse")
builder.add_edge("parse", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

final_state = graph.invoke({"user_query": "什么是LangGraph?"})   # 返回完整状态
```

**② 条件边 + 循环边**(分支和循环都是条件边,区别只在路由函数怎么返):

```python
def route_by_intent(state):
    return "summarize_node" if state.get("intent") == "summarize" else "rewrite_node"

builder.add_conditional_edges(
    "parse_intent",
    route_by_intent,
    {"summarize_node": "summarize_node", "rewrite_node": "rewrite_node"}   # 分支名→目标节点
)

# 循环边 = 条件边回环到前序节点,必须带终止条件(progress 到 100 就跳出)
def loop_router(state):
    return "final_node" if state["progress"] >= 100 else "loop_node"
```

**③ 并行执行**(扇出:一个节点分派多个;扇入:都完成后汇合):

```python
builder.add_edge("deduplicate", "summary")        # 扇出:同时激活 summary 和 keyword
builder.add_edge("deduplicate", "keyword")
builder.add_edge("summary", "sensitive_check")    # 扇入:两个并行节点都完成后才执行汇总
builder.add_edge("keyword", "sensitive_check")
```

**④ 状态持久化**(配合状态历史/断点续跑):

```python
from langgraph.checkpoint.memory import MemorySaver

app = builder.compile(checkpointer=MemorySaver())   # 开启状态存档
app.invoke(init_state, config={"configurable": {"thread_id": "test_001"}})  # thread_id=会话标识
```

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

### 核心代码模式:三种架构的骨架
**① 中心化 Supervisor**——主管用 LLM 决定"下一个派谁",条件边分发:

```python
class TaskState(TypedDict):
    task: str
    result: str
    next_agent: str        # Supervisor 的输出:下一个员工智能体的名字
    round_count: int

def supervisor_node(state):
    # 主管不干活,只负责拆任务、派活:让 LLM 输出"下一步去哪个员工"
    reply = llm.invoke(f"把任务拆分并指派给对应智能体,只输出智能体名:{state['task']}")
    return {"next_agent": reply.content, "round_count": state["round_count"] + 1}

builder.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],      # 路由函数直接返回 next_agent
    {"research": "research", "writer": "writer", "code": "code", ...}
)
# 注意:round_count 超过 MAX_ROUNDS 时强制走 end,防止无限分派
```

**② 链式 Sequence**——没有主管,固定边接力(本质就是第六章的线性图):

```python
builder.add_edge(START, "writer")          # 写 → 纠错 → 润色,前一个输出是后一个输入
builder.add_edge("writer", "corrector")
builder.add_edge("corrector", "polisher")
```

**③ 去中心化 Peer-to-peer**——平等智能体 + 条件边循环,直到任务完成:

```python
def should_continue(state):
    return "product" if not state["is_finished"] else "END"   # 运营干完看是否收工

builder.add_conditional_edges("operation", should_continue,
                              {"product": "product", "END": END})   # 没完成就回到产品智能体
```

### 7.2 复杂流程管控
- **子图 Subgraph**:图里嵌图(先编译子图,再当节点加进主图),实现逻辑隔离+复用
- **并行(扇出/扇入)**:一个节点分派多个智能体并行干 → 汇总节点整合;**并发冲突两种解法**:① 各并行节点用独立状态键(推荐);② 同字段用自定义合并函数
- **循环与重试**:审核不达标→回退重生成(条件边回环);**循环次数限制**防死循环

**核心代码模式**:

```python
# ① 子图:先编译,再当作"一个节点"塞进主图 → 逻辑隔离 + 可复用
subgraph = sub_builder.compile()
builder.add_node("grade", subgraph)          # 主图里这一站就是整个子流程

# ② 并行:扇出分派多个 → 扇入汇总(和第六章并行写法一致)
builder.add_edge("screening", "resume_agent")    # 扇出
builder.add_edge("screening", "skill_agent")
builder.add_edge("resume_agent", "summary")      # 扇入:都完成才汇总
builder.add_edge("skill_agent", "summary")
# 冲突处理:各并行节点写不同状态键(如 resume_info / skill_match),否则要自定义合并函数

# ③ 重试循环:审核 Agent 返回 pass/retry,retry 则回退生成节点;retry_count 限制次数
def review_router(state):
    if state["review_result"] == "pass" or state["retry_count"] >= 2:
        return "END"
    return "plot"        # 回环到生成节点重新写
```

### 7.3 人机协作(Human-in-the-loop)
- **检查点 MemorySaver**:每步状态存档(节点输出+state+下一节点),`thread_id`/session_id 恢复进度,崩溃续跑
- **中断 Interrupts**:`interrupt_before`(关键操作执行前人工授权,如发邮件/转账)、`interrupt_after`(执行后人工确认)
- **动态状态编辑**:`update_state()` 人工修改中间输出;或设置 `current_node` 把工作流**回退到任意节点**重新执行

**核心代码模式**:

```python
app = graph.compile(
    checkpointer=MemorySaver(),                    # 先开检查点,中断才有处可存
    interrupt_before=["send_email"],               # 发送邮件前停下,等人工授权
    # interrupt_after=["plot"]                     # 或执行后停下,等人工确认结果
)

result = app.invoke(init_state, config=config)     # 第一次运行:到 send_email 前中断
# ……人工确认……
app.invoke(None, config=config)                    # 再次调用:从断点继续执行

# 动态状态编辑:人工修改中间输出,或回退到任意节点
app.update_state(config, {"plot": "人工修改后的情节"})   # 改状态
app.update_state(config, {"current_node": "plot"})      # 回退:从 plot 节点重新跑
```

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

### 核心代码模式
**① 游戏状态**——所有游戏数据都在 State 里流转:

```python
class GameState(TypedDict):
    civilian_word: str               # 平民词
    undercover_word: str             # 卧底词
    role_assignment: dict            # {agent: (角色, 词语)}
    speeches: dict                   # 本轮发言
    history_speeches: List[Dict]     # 历史发言(多轮记忆)
    votes: dict                      # 本轮投票
    game_status: str                 # running / end
    winner: str                      # civilian / undercover
    eliminated: List[str]            # 已淘汰玩家
    round: int                       # 当前轮次
```

**② 图结构**——固定边串联流程,条件边实现"游戏循环"核心:

```python
builder = StateGraph(GameState)
builder.set_entry_point("generate_words")
builder.add_edge("generate_words", "assign_roles")       # 词语 → 角色
builder.add_edge("assign_roles", "generate_speeches")    # 角色 → 发言
builder.add_edge("generate_speeches", "vote_undercover") # 发言 → 投票
builder.add_edge("vote_undercover", "judge_result")      # 投票 → 裁决

def route(state: GameState):                             # 条件边=循环核心
    return "generate_speeches" if state["game_status"] == "running" else "show_final_result"

builder.add_conditional_edges("judge_result", route)     # 没结束→回到发言轮;结束→展示结果
builder.add_edge("show_final_result", END)
```

**③ 一个模块示例(发言生成)**——体现角色差异化策略 + 历史记忆 + JSON 输出约束:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是「谁是卧底」资深玩家,当前第{round}轮。
    规则:不能说出词语本身;结合历史发言避免重复/矛盾。
    - 平民:真实描述特征,帮同伴识别卧底
    - 卧底:模仿平民,模糊核心差异,不暴露身份
    输出JSON:{{"speech": "发言", "reason": "策略理由"}}"""),
    ("user", "你的角色是{role},拿到的词语是{word}")
])
chain = prompt | llm | parser            # 强制 JSON,再 json.loads 取 speech/reason
```

### 工程要点(面试可说)
- LLM 输出一律 JSON + try-except 兜底,保证程序稳定
- `history_speeches` 让智能体拥有跨轮记忆
- 提交规范:project 文件夹建个人目录(驼峰命名),GitHub PR 提交,含核心代码 + Readme,**禁止提交 .env**

---

## 整体知识体系(面试串联话术)

> 我可以先用 **LangChain** 的三大组件(模型调用、提示词模板、输出解析)搭出"输入可控、输出可预期"的基础应用;加上 **Memory 和 Tool** 让应用有记忆、能动手;面对复杂任务用**链式工作流(线性/路由/并行)+ 错误处理**拆解流程,用 **RAG** 让模型基于私有知识库回答;当流程复杂到需要分支、循环、多智能体协作时,升级到 **LangGraph**,用"状态-节点-边"建模,配合子图、并行、人机协作(中断/检查点)实现生产级系统;最后通过"谁是卧底"这类实战把全部知识串成完整的多智能体应用。
