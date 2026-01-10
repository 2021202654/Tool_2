from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # 引入占位符
from langchain.memory import ConversationBufferMemory # 引入内存模块
from graphene_tools import ml_prediction_tool, physics_calculation_tool

def build_agent(api_key, base_url, model_name):
    # 1. 配置 LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )

    # 2. 挂载工具
    tools = [ml_prediction_tool, physics_calculation_tool]

    # 3. 编写提示词 (Prompt)
    # 关键修改：加入了 chat_history 占位符，让 Agent 能看到之前的对话
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """
        你是专业的石墨烯科研助手。你擅长通过机器学习模型预测材料性质。
        
        【重要规则】
        1. 当用户询问预测时，必须提取以下三个参数：
           - 长度 (length/L)，单位 um，若未提供默认为 10.0
           - 温度 (temperature/T)，单位 K
           - 缺陷率 (defect)，范围 0-1
        2. 如果用户只更新了部分参数（例如“如果温度变成 400K 呢？”），请根据【对话历史】补全其他参数。
        3. 优先调用 `ml_prediction_tool`。
        4. 仅当预测值异常 (<10 或 >6000) 时，调用 `physics_calculation_tool` 核对。
        """),
        
        # === 这里存放历史对话记录 ===
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{input}"),
        
        # === 这里存放 Agent 的思考过程 ===
        ("placeholder", "{agent_scratchpad}"), 
    ])

    # 4. 组装 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. === 关键：初始化记忆模块 ===
    # memory_key 必须和上面的 prompt 变量名 "chat_history" 一致
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 6. 创建执行器 (注入 memory)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory 
    )
    
    return agent_executor