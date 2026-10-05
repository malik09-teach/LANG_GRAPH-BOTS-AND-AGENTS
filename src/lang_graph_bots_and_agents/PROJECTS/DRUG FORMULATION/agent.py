import os
import datetime
import requests
import operator
from typing import Annotated, List, TypedDict, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdShapeHelpers
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Fix for 400 Error: Use a model strictly optimized for tool calling on Groq
llm = ChatOllama(model="medgemma:4b")
# ==========================================
# CONSTANTS & REFERENCES
# ==========================================
E3_LIGANDS = {
    "Cereblon (CRBN)": "C1CC(=O)NC(=O)C1",
    "VHL": "C(=O)N1CC(C)CC1C(=O)NCC2=CC=C(C=C2)C3=CC=C(C=C3)C"
}

REF_SMILES = "c1ccccc1NC(=O)c2cnc(nc2)N" 
ref_mol = Chem.AddHs(Chem.MolFromSmiles(REF_SMILES))
AllChem.EmbedMolecule(ref_mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(ref_mol)

# ==========================================
# STATE & SCHEMAS
# ==========================================
class GenerationBatch(BaseModel):
    smiles_list: List[str] = Field(description="List of 3 valid PROTAC SMILES strings.")

class DrugDesignState(TypedDict):
    target_name: str
    thread_id: str
    auto_pdb_id: str
    auto_e3_ligase: str
    known_warheads: List[str]
    current_smiles: List[str]
    eval_results: Annotated[List[Dict[str, Any]], operator.add]
    best_candidates: Annotated[List[Dict[str, Any]], operator.add]
    iteration: int
    feedback: str
    history_log: Annotated[List[str], operator.add]

# ==========================================
# UTILITIES
# ==========================================
def write_report_event(thread_id: str, content: str, mode: str = "a"):
    os.makedirs("reports", exist_ok=True)
    filepath = os.path.abspath(os.path.join("reports", f"report_{thread_id}.txt"))
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return filepath

def fetch_pdb_id_automatically(target_name: str) -> str:
    try:
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": target_name}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 1}}
        }
        res = requests.post(url, json=query, timeout=5)
        if res.status_code == 200 and res.json().get("result_set"):
            return res.json()["result_set"][0]["identifier"]
    except Exception:
        pass
    return "1F86"

# ==========================================
# GRAPH NODES
# ==========================================
def ingest(state: DrugDesignState):
    """Initializes the design process."""
    return {"iteration": 0, "history_log": ["[Iteration 0] Ingested target specification."]}

def pre_search(state: DrugDesignState):
    """Live PDB and PubChem lookups."""
    target = state["target_name"]
    thread_id = state["thread_id"]
    
    pdb_id = fetch_pdb_id_automatically(target)
    e3_name = "VHL" if "kinase" in target.lower() else "Cereblon (CRBN)"
    
    known_refs = []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{target}/property/CanonicalSMILES/JSON"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            for prop in res.json().get("PropertyTable", {}).get("Properties", []):
                known_refs.append(prop.get("CanonicalSMILES"))
    except Exception:
        pass

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"--- DRUG DESIGN REPORT ---\nDate: {now_str}\nTarget Name: {target}\nAuto PDB: {pdb_id}\nAuto E3: {e3_name}\n"
    write_report_event(thread_id, header, mode="w")
    
    return {
        "auto_pdb_id": pdb_id,
        "auto_e3_ligase": e3_name,
        "known_warheads": known_refs,
        "history_log": ["[Pre-search] Completed target profiling."]
    }

def generate(state: DrugDesignState):
    """Generates SMILES using structured output."""
    iter_num = state.get("iteration", 0) + 1
    target = state["target_name"]
    e3_smiles = E3_LIGANDS[state.get("auto_e3_ligase", "Cereblon (CRBN)")]
    warheads = state.get("known_warheads", [])[:3]
    feedback = state.get("feedback", "Initial generation cycle.")
    
    prompt = f"""
    Design 3 chemically valid PROTAC SMILES for target: {target}.
    Known binder SMILES: {warheads if warheads else 'Create a novel core'}
    Previous Feedback: {feedback}
    MUST attach the warhead via a PEG or alkyl linker to this E3 anchor: {e3_smiles}
    Return ONLY valid SMILES strings.
    """
    
    res = llm.with_structured_output(GenerationBatch).invoke([HumanMessage(content=prompt)])
    return {
        "current_smiles": res.smiles_list, 
        "iteration": iter_num,
        "eval_results": [], # Reset parallel evaluation results list
        "history_log": [f"[Iteration {iter_num}] Generated 3 SMILES candidates."]
    }

def eval_props(state: DrugDesignState):
    """Calculates QED, LogP, and Molecular Weight."""
    results = []
    for smiles in state["current_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({"smiles": smiles, "source": "RDKit", "valid": False, "qed": 0.0, "logp": 0.0, "mw": 0.0})
            continue
        results.append({
            "smiles": smiles, "source": "RDKit", "valid": True,
            "qed": round(float(QED.qed(mol)), 3),
            "logp": round(float(Descriptors.MolLogP(mol)), 2),
            "mw": round(float(Descriptors.MolWt(mol)), 2)
        })
    return {"eval_results": results}

def eval_docking(state: DrugDesignState):
    """Calculates 3D shape similarity."""
    results = []
    for smiles in state["current_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({"smiles": smiles, "source": "Docking", "score": 0.0})
            continue
        try:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            shape_sim = 1.0 - rdShapeHelpers.ShapeTanimotoDist(ref_mol, mol_3d)
            score = round(-10.0 * shape_sim, 2)
            results.append({"smiles": smiles, "source": "Docking", "score": score})
        except Exception:
            results.append({"smiles": smiles, "source": "Docking", "score": -3.0})
    return {"eval_results": results}

def aggregate(state: DrugDesignState):
    """Combines parallel eval_props and eval_docking metrics."""
    smiles_map = {}
    for res in state["eval_results"]:
        s = res["smiles"]
        if s not in smiles_map:
            smiles_map[s] = {"smiles": s, "total_score": 0.0, "valid": False, "qed": 0.0, "logp": 0.0, "mw": 0.0, "docking": 0.0}

        if res["source"] == "RDKit":
            smiles_map[s].update({"valid": res["valid"], "qed": res["qed"], "logp": res["logp"], "mw": res["mw"]})
            smiles_map[s]["total_score"] += (res["qed"] * 20.0) if res["valid"] else -100.0
        elif res["source"] == "Docking":
            smiles_map[s]["docking"] = res["score"]
            smiles_map[s]["total_score"] += abs(res["score"]) * 2.0

    candidates = list(smiles_map.values())
    candidates.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    log_content = f"\n--- ITERATION {state['iteration']} AGGREGATED SCORES ---\n"
    for cand in candidates:
        log_content += f"SMILES: {cand['smiles']} | Score: {cand['total_score']}\n"
    write_report_event(state["thread_id"], log_content, mode="a")
    
    return {"best_candidates": [candidates[0]] if candidates else []}

def post_search(state: DrugDesignState):
    """Validates top SMILES existence in PubChem."""
    if not state.get("best_candidates"):
        return {}
    top_smi = state["best_candidates"][-1]["smiles"]
    
    exists = False
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{top_smi}/cids/JSON"
        if requests.get(url, timeout=3).status_code == 200:
            exists = True
    except Exception:
        pass
        
    val_content = f"POST-SEARCH: {top_smi} Exists in Literature: {exists}\n"
    write_report_event(state["thread_id"], val_content, mode="a")
    return {"history_log": [f"[Post-search] Checked PubChem presence: {exists}"]}

def critic(state: DrugDesignState):
    """Generates medicinal chemistry feedback for the next iteration."""
    latest = state["best_candidates"][-1] if state["best_candidates"] else None
    if not latest or not latest.get("valid", False):
        fb = "Previous candidate was chemically invalid. Produce a clean, valid SMILES core."
        return {"feedback": fb, "history_log": [f"[Critic Feedback] {fb}"]}

    prompt = f"""
    Current Candidate: {latest['smiles']}
    Molecular Weight: {latest['mw']} Da (Target: < 1000)
    QED Score: {latest['qed']}
    Shape Match Score: {latest['docking']} (Target: < -7.5)

    Give a concise 1-2 sentence medicinal chemistry critique explaining which chemical 
    group to modify to optimize the PROTAC drug-likeness.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    fb = response.content.strip()
    return {"feedback": fb, "history_log": [f"[Critic Feedback] {fb}"]}

def route_next_step(state: DrugDesignState) -> str:
    if state["iteration"] >= 4:
        return "finalize"
    if state["best_candidates"]:
        top = state["best_candidates"][-1]
        if top.get("valid") and top.get("qed", 0) >= 0.50 and top.get("docking", 0) <= -7.5:
            return "finalize"
    return "generate"

def finalize(state: DrugDesignState):
    """Completes the graph workflow."""
    return {"history_log": ["[Finalize] Design process complete."]}

# ==========================================
# GRAPH ARCHITECTURE MATCHING IMAGE
# ==========================================
builder = StateGraph(DrugDesignState)
builder.add_node("ingest", ingest)
builder.add_node("pre_search", pre_search)
builder.add_node("generate", generate)
builder.add_node("eval_docking", eval_docking)
builder.add_node("eval_props", eval_props)
builder.add_node("aggregate", aggregate)
builder.add_node("post_search", post_search)
builder.add_node("critic", critic)
builder.add_node("finalize", finalize)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "pre_search")
builder.add_edge("pre_search", "generate")
# Parallel evaluation branches
builder.add_edge("generate", "eval_docking")
builder.add_edge("generate", "eval_props")
builder.add_edge("eval_docking", "aggregate")
builder.add_edge("eval_props", "aggregate")
builder.add_edge("aggregate", "post_search")
builder.add_edge("post_search", "critic")
builder.add_conditional_edges("critic", route_next_step, {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

drug_agent = builder.compile(checkpointer=MemorySaver())