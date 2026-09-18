from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import os

# Import the compiled real-physics LangGraph agent from the agent.py file
from agent import drug_agent 

app = FastAPI(title="De Novo Drug Design Engine")

class DesignRequest(BaseModel):
    target_info: str
    thread_id: str

@app.post("/design-run")
def run_agent(request: DesignRequest):
    """Executes the agentic loop and saves a text report to disk."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # backend.py
    initial_state = {
           "target_info": request.target_info,
           "current_smiles": [],
           "eval_results": [],
           "best_candidates": [],
           "iteration": 0,
           "feedback": "Initialize generation based on target constraints.",
           "history_log": []  # <-- Add this line
                     }
    # Execute the graph
    print(f"Starting agent run for thread: {request.thread_id}...")
    final_state = drug_agent.invoke(initial_state, config=config)
    
    candidates = final_state.get("best_candidates", [])
    
    # ==========================================
    # TEXT REPORT GENERATION
    # ==========================================
    report_filename = f"{request.thread_id}_design_report.txt"
    
    with open(report_filename, "w") as f:
        f.write(f"--- DRUG DESIGN REPORT ---\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session ID: {request.thread_id}\n")
        f.write(f"Target Constraints:\n{request.target_info}\n")
        f.write("-" * 40 + "\n")
        
        if candidates:
            f.write("FINAL CANDIDATES:\n\n")
            # Sort candidates by total_score descending
            sorted_cands = sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)
            
            for idx, cand in enumerate(sorted_cands, 1):
                f.write(f"Candidate #{idx}\n")
                f.write(f"SMILES:  {cand.get('smiles', 'N/A')}\n")
                f.write(f"Valid:   {cand.get('valid', False)}\n")
                f.write(f"QED:     {cand.get('qed', 'N/A')}\n")
                f.write(f"LogP:    {cand.get('logp', 'N/A')}\n")
                f.write(f"MW:      {cand.get('mw', 'N/A')} Da\n")
                f.write(f"Docking: {cand.get('docking', 'N/A')} kcal/mol\n")
                f.write(f"Score:   {cand.get('total_score', 'N/A')}\n")
                f.write("\n")
        else:
            f.write("Status: Agent failed to generate valid candidates within iteration limits.\n")
            
    print(f"Report saved to {os.path.abspath(report_filename)}")
    
    return {"candidates": candidates, "report_file": report_filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)