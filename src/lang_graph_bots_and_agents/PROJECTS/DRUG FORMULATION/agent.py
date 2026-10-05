import os
import datetime
import requests
import operator
import re
import json
from typing import Annotated, List, TypedDict, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdShapeHelpers
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

# LLM Instance
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

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
# CUSTOM TOOLS & UTILITIES
# ==========================================
def write_report_event(thread_id: str, content: str, mode: str = "a"):
    os.makedirs("reports", exist_ok=True)
    filepath = os.path.abspath(os.path.join("reports", f"report_{thread_id}.txt"))
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return filepath

@tool
def fetch_pdb_tool(target_name: str) -> str:
    """Queries RCSB PDB API for a given target name."""
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

@tool
def fetch_pubchem_tool(target_name: str) -> List[str]:
    """Retrieves canonical SMILES for target binders from PubChem."""
    known_refs = []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{target_name}/property/CanonicalSMILES/JSON"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            for prop in res.json().get("PropertyTable", {}).get("Properties", []):
                smi = prop.get("CanonicalSMILES")
                if smi:
                    known_refs.append(smi)
    except Exception:
        pass
    return known_refs

# Registry for manual tool calling
TOOL_REGISTRY = {
    "fetch_pdb_tool": fetch_pdb_tool,
    "fetch_pubchem_tool": fetch_pubchem_tool
}

# ==========================================
# GRAPH NODES
# ==========================================
def ingest(state: DrugDesignState):
    """Initializes the design process."""
    return {"iteration": 0, "history_log": ["[Iteration 0] Ingested target specification."]}

def pre_search_custom_node(state: DrugDesignState):
    """Custom Node: Executes tool functions programmatically without relying on LLM auto-invocation."""
    target = state["target_name"]
    thread_id = state["thread_id"]
    
    # Direct execution of tools via custom node logic
    pdb_id = fetch_pdb_tool.invoke({"target_name": target})
    known_refs = fetch_pubchem_tool.invoke({"target_name": target})
    
    e3_name = "VHL" if "kinase" in target.lower() else "Cereblon (CRBN)"

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"--- DRUG DESIGN REPORT ---\nDate: {now_str}\nTarget Name: {target}\nAuto PDB: {pdb_id}\nAuto E3: {e3_name}\n"
    write_report_event(thread_id, header, mode="w")
    
    return {
        "auto_pdb_id": pdb_id,
        "auto_e3_ligase": e3_name,
        "known_warheads": known_refs,
        "history_log": ["[Pre-search] Completed target profiling via custom tool node."]
    }

def generate_custom_node(state: DrugDesignState):
    """Generates SMILES with dual fallback (Structured Output -> Manual Regex/JSON Extraction)."""
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
    
    Format your output strictly as a JSON object with key "smiles_list":
    {{"smiles_list": ["SMILES1", "SMILES2", "SMILES3"]}}
    """
    
    smiles = []
    try:
        # Try structured output first
        res = llm.with_structured_output(GenerationBatch).invoke([HumanMessage(content=prompt)])
        smiles = res.smiles_list
    except Exception:
        # Custom fallback node logic if model tool-calling fails
        raw_res = llm.invoke([HumanMessage(content=prompt)]).content
        
        # Try extracting JSON manually
        json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                smiles = data.get("smiles_list", [])
            except Exception:
                pass
        
        # Fallback regex search for valid SMILES strings if JSON fails
        if not smiles:
            candidates = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)\\\/=#$%]{10,}', raw_res)
            smiles = [c for c in candidates if Chem.MolFromSmiles(c) is not None][:3]

    return {
        "current_smiles": smiles, 
        "iteration": iter_num,
        "eval_results": [],
        "history_log": [f"[Iteration {iter_num}] Generated {len(smiles)} SMILES candidates."]
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
# GRAPH ARCHITECTURE
# ==========================================
builder = StateGraph(DrugDesignState)
builder.add_node("ingest", ingest)
builder.add_node("pre_search", pre_search_custom_node)
builder.add_node("generate", generate_custom_node)
builder.add_node("eval_docking", eval_docking)
builder.add_node("eval_props", eval_props)
builder.add_node("aggregate", aggregate)
builder.add_node("post_search", post_search)
builder.add_node("critic", critic)
builder.add_node("finalize", finalize)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "pre_search")
builder.add_edge("pre_search", "generate")
builder.add_edge("generate", "eval_docking")
builder.add_edge("generate", "eval_props")
builder.add_edge("eval_docking", "aggregate")
builder.add_edge("eval_props", "aggregate")
builder.add_edge("aggregate", "post_search")
builder.add_edge("post_search", "critic")
builder.add_conditional_edges("critic", route_next_step, {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

drug_agent = builder.compile(checkpointer=MemorySaver())