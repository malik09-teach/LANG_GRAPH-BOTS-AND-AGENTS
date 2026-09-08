from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from agents.supervisor import AgentSupervisor

app = FastAPI(title="Interactive Agentic Formulation API")

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for the active session (for demo purposes)
class SessionState:
    def __init__(self):
        self.variables = []
        self.constraints = ""
        self.history = []
        self.supervisor = None

session = SessionState()

# Pydantic Models
class Variable(BaseModel):
    name: str
    symbol: str
    min: float
    max: float

class InitRequest(BaseModel):
    variables: List[Variable]
    constraints: str

class ReportRequest(BaseModel):
    candidate: List[float]
    score: float

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/init")
def initialize_session(req: InitRequest):
    session.variables = [v.dict() for v in req.variables]
    session.constraints = req.constraints
    session.history = []
    session.supervisor = AgentSupervisor(session.variables, session.constraints)
    return {"status": "initialized", "variables": session.variables}

@app.get("/api/propose")
def get_proposal():
    if not session.supervisor:
        return {"error": "Session not initialized"}
        
    proposal = session.supervisor.get_next_proposal(session.history)
    return {
        "candidate": proposal["candidate"],
        "reason": proposal["reason"],
        "iteration": len(session.history) + 1
    }

@app.post("/api/report")
def report_result(req: ReportRequest):
    if not session.supervisor:
        return {"error": "Session not initialized"}
        
    session.history.append({
        "candidate": req.candidate,
        "score": req.score
    })
    
    return {"status": "recorded", "history_length": len(session.history)}

@app.get("/api/state")
def get_state():
    return {
        "variables": session.variables,
        "constraints": session.constraints,
        "history": session.history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
