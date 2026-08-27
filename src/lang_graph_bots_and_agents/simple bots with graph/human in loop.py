import os
import json
import urllib.request
import urllib.parse
import streamlit as st
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# --- 1. Secure Environment Setup ---
st.set_page_config(page_title="Secure ReAct Agent", layout="wide")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

# --- 2. Live Tools Setup ---
@tool
def get_live_weather(city: str) -> str:
    """Fetch real-time weather data for a specific city name."""
    encoded_city = urllib.parse.quote(city)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    try:
        with urllib.request.urlopen(geo_url) as geo_response:
            geo_data = json.loads(geo_response.read().decode())
            if not geo_data.get("results"):
                return f"Could not find coordinates for city: {city}"
            location = geo_data["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            resolved_city = location["name"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(weather_url) as weather_response:
            weather_data = json.loads(weather_response.read().decode())
            current = weather_data["current_weather"]
            return f"The current temperature in {resolved_city} is {current['temperature']}°C with wind speed {current['windspeed']} km/h."
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool
def search_arxiv(query: str) -> str:
    """Search the ArXiv academic database for scientific papers across all disciplines.
    
    CRITICAL INSTRUCTION: The query must be STRICTLY keywords or use ArXiv field prefixes. 
    DO NOT use conversational language like 'find papers about'.
    
    Valid query formats:
    - Broad keyword search: 'large language models', 'quantum computing', 'gene editing'
    - Specific fields: 'ti:transformer AND au:vaswani' (Title and Author)
    - Abstract only: 'abs:reinforcement learning'
    - All fields: 'all:computer vision'
    """
    try:
        # Increased top_k_results to retrieve a wider variety of papers
        arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1500)
        result = arxiv.run(query)
        
        # Guide the LLM to self-correct if the search yields nothing
        if not result or "No good Arxiv Result" in result:
            return f"Search failed. The query '{query}' returned no results. Try again using broader keywords or checking your syntax."
            
        return result
    except Exception as e:
        return f"ArXiv API Error: {str(e)}"

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
)

tools = [wikipedia_tool, search_arxiv, get_live_weather]

# --- 3. Robust Graph Construction ---
@st.cache_resource
def build_agent():
    # Utilizing the Qwen multimodal model supported natively by Groq
    llm = ChatGroq(model="qwen/qwen3.6-27b", groq_api_key=GROQ_API_KEY)
    t_llm = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def botnode(state: MessagesState):
        response = t_llm.invoke(state["messages"])
        return {"messages": [response]}

    def approve_execution(state: MessagesState):
        last_message = state["messages"][-1]
        decision = interrupt({
            "type": "pre_execution",
            "title": "Approval Required: Tool Execution",
            "tool_calls": last_message.tool_calls
        })

        if not decision.get("approved", False):
            rejections = [
                ToolMessage(
                    tool_call_id=call["id"],
                    content=f"Action denied by human researcher: {decision.get('reason', 'Rejected')}"
                )
                for call in last_message.tool_calls
            ]
            return {"messages": rejections}
        return {}

    def approve_result(state: MessagesState):
        last_message = state["messages"][-1]
        decision = interrupt({
            "type": "post_execution",
            "title": "Review Required: Tool Results",
            "raw_result": last_message.content
        })

        if decision.get("edited_result"):
            # Properly overwrite the state by generating a new object with the identical LangChain ID
            edited_msg = ToolMessage(
                content=decision["edited_result"],
                tool_call_id=last_message.tool_call_id,
                id=last_message.id 
            )
            return {"messages": [edited_msg]}
        return {}

    def route_after_approval(state: MessagesState):
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            return "botnode"
        return "tools"

    workflow = StateGraph(MessagesState)
    workflow.add_node("botnode", botnode)
    workflow.add_node("approve_execution", approve_execution)
    workflow.add_node("tools", tool_node)
    workflow.add_node("approve_result", approve_result)

    workflow.add_edge(START, "botnode")
    workflow.add_conditional_edges("botnode", tools_condition, {"tools": "approve_execution", "__end__": END})
    workflow.add_conditional_edges("approve_execution", route_after_approval, {"botnode": "botnode", "tools": "tools"})
    workflow.add_edge("tools", "approve_result")
    workflow.add_edge("approve_result", "botnode")

    return workflow.compile(checkpointer=MemorySaver())

app = build_agent()

# --- 4. Streamlit UI & State Syncing ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = os.urandom(8).hex()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🤖 Secure ReAct Agent (Human-in-the-Loop)")
st.caption(f"Session Thread ID: `{st.session_state.thread_id}` | Keys secured internally.")

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.thread_id = os.urandom(8).hex()
    st.rerun()

# Sync UI strictly to the LangGraph backend state
state = app.get_state(config)
messages = state.values.get("messages", [])

for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if msg.content:
                st.markdown(msg.content)
            if msg.tool_calls:
                with st.expander("🛠️ Tool Call Requested", expanded=False):
                    st.json(msg.tool_calls)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool", avatar="🔧"):
            with st.expander(f"Data from Tool (ID: {msg.tool_call_id[:5]}...)", expanded=False):
                st.code(msg.content, language="text")

# --- 5. Interruption Handling ---
is_paused = len(state.next) > 0

if is_paused:
    interrupt_task = state.tasks[0]
    if interrupt_task.interrupts:
        interrupt_data = interrupt_task.interrupts[0].value
        
        st.warning("⚠️ **Human Review Gate Triggered**")
        with st.container(border=True):
            st.subheader(interrupt_data.get("title", "Approval Required"))
            
            if interrupt_data.get("type") == "pre_execution":
                st.markdown("The agent requested the following action:")
                st.json(interrupt_data.get("tool_calls", []))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve Tool"):
                        app.invoke(Command(resume={"approved": True}), config=config)
                        st.rerun()
                with col2:
                    if st.button("❌ Deny Tool"):
                        app.invoke(Command(resume={"approved": False, "reason": "User denied."}), config=config)
                        st.rerun()

            elif interrupt_data.get("type") == "post_execution":
                st.markdown("Raw data returned from the tool:")
                raw_text = interrupt_data.get("raw_result", "")
                edited_text = st.text_area("Edit Tool Output (Optional)", value=raw_text, height=150)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Accept As-Is"):
                        app.invoke(Command(resume={"edited_result": None}), config=config)
                        st.rerun()
                with col2:
                    if st.button("✏️ Submit Edited Data"):
                        app.invoke(Command(resume={"edited_result": edited_text}), config=config)
                        st.rerun()

# --- 6. Chat Input ---
if not is_paused:
    user_input = st.chat_input("Ask a question to trigger the workflow...")
    if user_input:
        # Directly invoke the graph; it will execute until it finishes or hits a new interrupt
        app.invoke({"messages": [("user", user_input)]}, config=config)
        st.rerun()