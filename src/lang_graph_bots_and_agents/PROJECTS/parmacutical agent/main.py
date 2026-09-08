from fastapi import FastAPI
from agents.supervisor import AgentSupervisor

app = FastAPI(title="Neuro-Symbolic Agentic Formulation API")

@app.get("/")
def read_root():
    return {"message": "Agentic Formulation System is running."}

@app.post("/optimize")
def run_optimization_cycle(iterations: int = 5):
    """
    Run a full optimization loop for a given number of iterations.
    """
    supervisor = AgentSupervisor()
    final_state = supervisor.run_optimization(iterations)
    
    return {
        "status": "success",
        "total_iterations_run": final_state["iteration"],
        "best_score": max([h["score"] for h in final_state["history"]]) if final_state["history"] else 0.0,
        "history": final_state["history"]
    }

if __name__ == "__main__":
    import uvicorn
    # For local testing without full API setup, we can also just run it as a script:
    print("Initializing Neuro-Symbolic Agentic Formulation System...")
    supervisor = AgentSupervisor()
    print("Starting 10-iteration optimization loop...")
    final_state = supervisor.run_optimization(10)
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    scores = [h["score"] for h in final_state["history"]]
    best_score = max(scores)
    best_idx = scores.index(best_score)
    print(f"Best Formulation: {final_state['history'][best_idx]['candidate']}")
    print(f"Best Stability Score: {best_score:.2f}")
