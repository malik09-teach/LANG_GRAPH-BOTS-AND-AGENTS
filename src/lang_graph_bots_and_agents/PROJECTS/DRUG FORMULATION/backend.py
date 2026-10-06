import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import drug_agent 

app = FastAPI(title="Dynamic PROTAC Backend")

# Allow cross-origin requests from Streamlit or any local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TargetPayload(BaseModel):
    target_info: str
    thread_id: str

@app.post("/design-run")
def run_design(data: TargetPayload):
    # Extract clean target name from input string
    raw_target = data.target_info.replace("Target:", "").split("\n")[0].strip()
    target_name = raw_target if raw_target else "BRD4"
    
    initial_input = {
        "target_name": target_name,
        "thread_id": data.thread_id,
        "iteration": 0,
        "history_log": [] 
    }
    
    config = {"configurable": {"thread_id": data.thread_id}}
    
    try:
        result = drug_agent.invoke(initial_input, config=config)
        report_path = os.path.abspath(os.path.join("reports", f"report_{data.thread_id}.txt"))
        
        # Safely extract candidates and best_candidate
        best_candidate = result.get("best_candidate") or {}
        final_candidates = result.get("eval_results") or []
        
        return {
            "status": "success",
            "thread_id": data.thread_id,
            "target_name": target_name,
            "auto_pdb_id": result.get("auto_pdb_id", "N/A"),
            "auto_e3_ligase": result.get("auto_e3_ligase", "N/A"),
            "report_file": report_path,
            "best_candidate": best_candidate,
            "final_candidates": final_candidates,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{thread_id}")
def get_report(thread_id: str):
    filepath = os.path.abspath(os.path.join("reports", f"report_{thread_id}.txt"))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report file not found.")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {"thread_id": thread_id, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)