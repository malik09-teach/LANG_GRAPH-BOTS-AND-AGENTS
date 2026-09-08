from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from engine.bayesian_optimizer import BayesianOptimizer
from agents.constraint_validator import ConstraintValidatorAgent
from rag.knowledge_base import ScientificKnowledgeBase
from simulation.lab_simulator import LabSimulator

# Define the state for LangGraph
class GraphState(TypedDict):
    iteration: int
    max_iterations: int
    candidates: List[List[float]]
    selected_candidate: List[float]
    experiment_result: float
    history: List[Dict[str, Any]]

class AgentSupervisor:
    def __init__(self):
        self.optimizer = BayesianOptimizer()
        self.knowledge_base = ScientificKnowledgeBase()
        self.validator = ConstraintValidatorAgent(self.knowledge_base)
        self.simulator = LabSimulator()
        
        # Bounds: [excipient_a, excipient_b, temp]
        self.bounds = [(0.0, 1.0), (0.0, 1.0), (10.0, 80.0)]
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        # Define nodes
        workflow.add_node("propose_candidates", self.propose_candidates)
        workflow.add_node("validate_and_select", self.validate_and_select)
        workflow.add_node("execute_lab", self.execute_lab)
        
        # Set entry point
        workflow.set_entry_point("propose_candidates")
        
        # Define edges
        workflow.add_edge("propose_candidates", "validate_and_select")
        workflow.add_edge("validate_and_select", "execute_lab")
        
        # Conditional exit
        workflow.add_conditional_edges(
            "execute_lab",
            self.should_continue,
            {
                "continue": "propose_candidates",
                "end": END
            }
        )
        
        return workflow.compile()
        
    def propose_candidates(self, state: GraphState):
        """Node 1: Bayesian Optimizer proposes candidates."""
        print(f"\n--- Iteration {state['iteration']} ---")
        print("BO Engine: Proposing candidates...")
        candidates = self.optimizer.suggest_next_experiment(bounds=self.bounds, num_candidates=5)
        return {"candidates": candidates}

    def validate_and_select(self, state: GraphState):
        """Node 2: LLM Agent validates candidates using RAG."""
        candidates = state["candidates"]
        selected = None
        
        print("Constraint Agent: Validating proposals against scientific RAG...")
        for cand in candidates:
            is_valid, reason = self.validator.validate(cand)
            if is_valid:
                selected = cand
                print(f" -> Selected: {cand} (Reason: {reason})")
                break
            else:
                print(f" -> Rejected: {cand} (Reason: {reason})")
                
        # Fallback if all rejected
        if selected is None:
            print(" -> All rejected by semantic rules. Proceeding with first candidate anyway for data collection.")
            selected = candidates[0]
            
        return {"selected_candidate": selected}

    def execute_lab(self, state: GraphState):
        """Node 3: Execute in physical lab simulator and update BO."""
        cand = state["selected_candidate"]
        print("Lab Simulator: Executing experiment...")
        score = self.simulator.execute_experiment(*cand)
        print(f"Lab Result: Stability Score = {score:.2f}")
        
        # Update the optimizer
        self.optimizer.update_data(cand, score)
        
        # Update history
        history = state.get("history", [])
        history.append({"candidate": cand, "score": score})
        
        return {"experiment_result": score, "history": history, "iteration": state["iteration"] + 1}
        
    def should_continue(self, state: GraphState):
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "continue"
        
    def run_optimization(self, iterations: int):
        initial_state = {
            "iteration": 0,
            "max_iterations": iterations,
            "candidates": [],
            "selected_candidate": [],
            "experiment_result": 0.0,
            "history": []
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state
