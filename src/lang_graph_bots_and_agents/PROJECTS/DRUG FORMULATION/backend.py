import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import drug_agent

app = FastAPI(title="Dynamic PROTAC Backend")

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
        "auto_pdb_id": "",
        "auto_e3_ligase": "",
        "known_warheads": [],
        "current_candidates": [],
        "best_seeds": [],
        "iteration": 0
    }
    
    config = {"configurable": {"thread_id": data.thread_id}}
    try:
        result = drug_agent.invoke(initial_input, config=config)
        report_path = os.path.abspath(os.path.join("reports", f"report_{data.thread_id}.txt"))
        
        return {
            "status": "success",
            "thread_id": data.thread_id,
            "auto_pdb_id": result.get("auto_pdb_id"),
            "auto_e3_ligase": result.get("auto_e3_ligase"),
            "report_file": report_path,
            "candidates": result.get("current_candidates", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{thread_id}")
def get_report(thread_id: str):
    filepath = os.path.abspath(os.path.join("reports", f"report_{thread_id}.txt"))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report file not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return {"content": f.read()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000)