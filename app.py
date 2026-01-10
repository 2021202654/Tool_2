import streamlit as st
# 引入我们写好的构建函数
from graphene_agent import build_agent 

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="石墨烯热导率预测助手", 
    page_icon="🧪", 
    layout="wide"
)

st.title("🧪 石墨烯科研助手 (Graphene Agent)")
st.caption("基于 XGBoost 机器学习模型与 K-C 物理理论的混合专家系统")

# --- 2. 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 获取 API 配置
    api_key = st.text_input("输入 API Key", type="password", help="请输入你的豆包/OpenAI API Key")
    base_url = st.text_input("Base URL", value="https://ark.cn-beijing.volces.com/api/v3")
    model_name = st.text_input("模型名称", value="doubao-seed-1-6-251015") 
    
    st.divider()
    
    # 清空历史按钮
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        # 清除缓存的 Agent，确保参数变更后能重新加载
        st.cache_resource.clear()
        st.rerun()

# --- 3. 初始化 Session State (对话历史) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是石墨烯科研助手。我可以帮你预测材料热导率。\n试试问我：预测一下 300K 温度下，缺陷为 0.5% 的石墨烯热导率。"}
    ]

# --- 4. 【关键修改】定义带缓存的 Agent 获取函数 ---
@st.cache_resource(show_spinner=False)
def get_agent_executor(api_key, base_url, model_name):
    """
    使用 st.cache_resource 缓存 Agent 对象。
    只有当 api_key, base_url 或 model_name 发生变化时，
    才会重新执行 build_agent，否则直接返回内存中的对象。
    """
    print("--- 正在初始化新的 Agent 实例 ---") # 调试用，方便在终端看到何时重建了
    return build_agent(api_key, base_url, model_name)

# --- 5. 渲染历史消息 ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 6. 处理用户输入 ---
if prompt_input := st.chat_input("请输入你的科研问题..."):
    # 6.1 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)

    # 6.2 检查 Key 是否存在
    if not api_key:
        st.warning("⚠️ 请先在左侧侧边栏输入 API Key！")
        st.stop()

    # 6.3 Agent 回复
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent 正在思考并调用工具..."):
                # === 修改点：使用缓存函数获取 executor ===
                # 即使循环调用，只要参数没变，这里瞬间就能拿到对象
                executor = get_agent_executor(api_key, base_url, model_name)
                
                # 调用 Agent (新版 LangChain 必须传字典)
                # 注意：这里我们还没有加记忆功能，下一阶段修改 graphene_agent.py 时会加上
                response = executor.invoke({"input": prompt_input})
                
                output_text = response["output"]
                st.write(output_text)
                
            # 保存助手回复到历史
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.markdown("建议检查：API Key 是否有效，或模型名称是否正确。")
            # 如果出错，可能是连接断了，清除缓存以便下次重试
            st.cache_resource.clear()