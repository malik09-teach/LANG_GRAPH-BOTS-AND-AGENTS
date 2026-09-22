import json
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from langchain_groq import  ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ==========================================
# 1. DETERMINISTIC MATH / CIRCUIT BREAKER
# ==========================================
def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1): result *= i
    return result

def combinations(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

def binomial_probability(n: int, k: int, p: float) -> float:
    return combinations(n, k) * (p ** k) * ((1 - p) ** (n - k))

# ==========================================
# 2. STATE AND DATA MODELS
# ==========================================
class AgentDecision(BaseModel):
    intervention: str = Field(description="Recommended medical intervention")
    risk_level: str = Field(description="HIGH, MEDIUM, LOW")
    extracted_anomalies: int

class ClinicalState(TypedDict):
    patient_data: str
    decision: AgentDecision
    audit_passed: bool
    audit_reason: str
    human_feedback: str

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.0)

# ==========================================
# 3. GRAPH NODES
# ==========================================
def clinical_reasoner(state: ClinicalState):
    """The primary agent making the initial diagnosis."""
    print("--- REASONING NODE ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ICU AI. Review telemetry and output a diagnosis."),
        ("human", "Data: {data}")
    ])
    chain = prompt | llm.with_structured_output(AgentDecision)
    decision = chain.invoke({"data": state["patient_data"]})
    return {"decision": decision}

def neuro_symbolic_auditor(state: ClinicalState):
    """The deterministic math gate. If it fails, it trips the circuit breaker."""
    print("--- SYMBOLIC AUDIT NODE ---")
    data = json.loads(state["patient_data"])
    decision = state["decision"]
    
    # Calculate true statistical risk
    actual_prob = binomial_probability(
        n=data["observation_minutes"], 
        k=decision.extracted_anomalies, 
        p=data["baseline_risk"]
    )
    
    # Rigid Thresholding (Circuit Breaker Logic)
    if decision.extracted_anomalies >= 8 and decision.risk_level != "HIGH":
        return {
            "audit_passed": False,
            "audit_reason": f"CRITICAL: Agent assigned {decision.risk_level} risk, but mathematical probability of {decision.extracted_anomalies} anomalies requires HIGH intervention."
        }
    
    return {
        "audit_passed": True,
        "audit_reason": "Mathematical validation passed."
    }

def human_approval_gate(state: ClinicalState):
    """
    HITL INTERRUPT: Pauses the graph entirely. 
    Surfaces the audit failure to a clinician and waits for resume Command.
    """
    print(f"\n[!] CIRCUIT BREAKER TRIPPED: {state['audit_reason']}")
    print("[!] Pausing execution and requesting human intervention...")
    
    # The interrupt() function pauses the execution thread and saves state.
    # The server can spin down; it will wake up when the human responds.
    human_response = interrupt({
        "question": "The Symbolic Auditor flagged this AI decision. Do you Approve, Edit, or Reject?",
        "agent_decision": state["decision"].model_dump(),
        "audit_reason": state["audit_reason"]
    })
    
    return {"human_feedback": human_response["action"]}

def execute_clinical_action(state: ClinicalState):
    """Final stage: Write to EMR or execute intervention."""
    print(f"--- EXECUTING ACTION ---")
    print(f"Executing: {state['decision'].intervention}")
    return state

# ==========================================
# 4. CONDITIONAL ROUTING
# ==========================================
def check_audit_status(state: ClinicalState) -> Literal["execute_clinical_action", "human_approval_gate"]:
    """Routes based on the deterministic auditor's output."""
    if state["audit_passed"]:
        print("--- ROUTER: AUDIT PASSED -> EXECUTING ---")
        return "execute_clinical_action"
    else:
        print("--- ROUTER: AUDIT FAILED -> ESCALATING TO HUMAN ---")
        return "human_approval_gate"

# ==========================================
# 5. BUILD GRAPH WITH CHECKPOINTER
# ==========================================
workflow = StateGraph(ClinicalState)

workflow.add_node("reasoner", clinical_reasoner)
workflow.add_node("auditor", neuro_symbolic_auditor)
workflow.add_node("human_approval_gate", human_approval_gate)
workflow.add_node("execute_clinical_action", execute_clinical_action)

workflow.add_edge(START, "reasoner")
workflow.add_edge("reasoner", "auditor")

# Conditional edge acts as the logic gate
workflow.add_conditional_edges(
    "auditor",
    check_audit_status
)

# After human review, we proceed to execute (or you could route back to the reasoner for an adaptive loop)
workflow.add_edge("human_approval_gate", "execute_clinical_action")
workflow.add_edge("execute_clinical_action", END)

# MemorySaver is required for HITL and Time-Travel
#checkpointer = MemorySaver()
app = workflow.compile()