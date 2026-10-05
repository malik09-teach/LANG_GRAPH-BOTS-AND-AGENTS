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
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdShapeHelpers
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load Environment Variables (Ensure GROQ_API_KEY is in your .env)
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b")

# ==========================================
# CONSTANTS & REFERENCES
# ==========================================
E3_LIGANDS = {
    "Cereblon (CRBN)": "C1CC(=O)NC(=O)C1",
    "VHL": "C(=O)N1CC(C)CC1C(=O)NCC2=CC=C(C=C2)C3=CC=C(C=C3)C"
}

# Reference binder for 3D shape comparison
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
    eval_results: List[Dict[str, Any]]
    best_candidate: Dict[str, Any]
    iteration: int
    feedback: str
    history_log: Annotated[List[str], operator.add]

# ==========================================
# CUSTOM TOOLS & UTILITIES
# ==========================================
def write_report_event(thread_id: str, content: str, mode: str = "a", title: str = None):
    os.makedirs("reports", exist_ok=True)
    filepath = os.path.abspath(os.path.join("reports", f"report_{thread_id}.txt"))
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_entry = f"[{now_str}]\n"
    if title:
        formatted_entry += f"{'='*60}\n 🔬 {title.upper()} \n{'='*60}\n"
    
    formatted_entry += f"{content}\n"
    if title:
        formatted_entry += f"{'-'*60}\n\n"
    else:
        formatted_entry += "\n"

    with open(filepath, mode, encoding="utf-8") as f:
        f.write(formatted_entry)
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
def fetch_pubchem_tool(compound_name: str) -> List[str]:
    """Retrieves canonical SMILES for known binder compounds from PubChem."""
    known_refs = []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
        res = requests.get(url, timeout=5)
        res.raise_for_status() 
        for prop in res.json().get("PropertyTable", {}).get("Properties", []):
            if smi := prop.get("CanonicalSMILES"):
                known_refs.append(smi)
    except Exception as e:
        print(f"PubChem Tool Error for {compound_name}: {e}")
    return known_refs

# ==========================================
# GRAPH NODES
# ==========================================
def ingest(state: DrugDesignState):
    write_report_event(state["thread_id"], "Agent Initialization Complete.", mode="w", title="SYSTEM START")
    return {"iteration": 0, "history_log": ["[Iteration 0] Ingested target specification."]}

def pre_search_custom_node(state: DrugDesignState):
    """Executes tools and auto-discovers known inhibitors autonomously."""
    target = state["target_name"]
    thread_id = state["thread_id"]
    
    pdb_id = fetch_pdb_tool.invoke({"target_name": target})
    
    # AUTONOMOUS LOGIC: Ask LLM to identify the best known inhibitor
    discovery_prompt = f"What is a highly well-known, FDA-approved small molecule inhibitor for the protein target '{target}'? Reply ONLY with the drug name (e.g., Gefitinib, Imatinib, JQ1) and absolutely nothing else."
    
    try:
        discovered_compound = llm.invoke([HumanMessage(content=discovery_prompt)]).content.strip()
        discovered_compound = re.sub(r'[^a-zA-Z0-9\-\s]', '', discovered_compound)
    except Exception:
        discovered_compound = target 
        
    known_refs = fetch_pubchem_tool.invoke({"compound_name": discovered_compound})
    e3_name = "VHL" if "kinase" in target.lower() else "Cereblon (CRBN)"

    info = (
        f"Target Name: {target}\n"
        f"Auto-Discovered Warhead: {discovered_compound}\n"
        f"Auto PDB ID: {pdb_id}\n"
        f"Auto E3 Ligase: {e3_name}\n"
        f"Found {len(known_refs)} known warhead SMILES."
    )
    write_report_event(thread_id, info, title="TARGET PROFILING")
    
    return {
        "auto_pdb_id": pdb_id,
        "auto_e3_ligase": e3_name,
        "known_warheads": known_refs,
        "history_log": [f"[Pre-search] Auto-discovered {discovered_compound}."]
    }

def generate_custom_node(state: DrugDesignState):
    """Generates SMILES with dual fallback and strict space removal."""
    iter_num = state.get("iteration", 0) + 1
    target = state["target_name"]
    e3_smiles = E3_LIGANDS[state.get("auto_e3_ligase", "Cereblon (CRBN)")]
    warheads = state.get("known_warheads", [])[:2]
    feedback = state.get("feedback", "Initial generation cycle. Design a PROTAC connecting the warhead to the E3 ligase via a PEG/alkyl linker.")
    
    prompt = f"""
    Design 3 chemically valid PROTAC SMILES for target: {target}.
    Known binder SMILES to use as warhead: {warheads if warheads else 'Create a novel core'}
    Previous Feedback: {feedback}
    MUST attach the warhead via a PEG or alkyl linker to this E3 anchor: {e3_smiles}
    
    Format your output strictly as a JSON object with key "smiles_list":
    {{"smiles_list": ["SMILES1", "SMILES2", "SMILES3"]}}
    """
    
    smiles = []
    try:
        res = llm.with_structured_output(GenerationBatch).invoke([HumanMessage(content=prompt)])
        smiles = res.smiles_list
    except Exception:
        raw_res = llm.invoke([HumanMessage(content=prompt)]).content
        json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                smiles = data.get("smiles_list", [])
            except Exception:
                pass
        
        if not smiles:
            candidates = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)\\\/=#$%]{10,}', raw_res)
            smiles = [c for c in candidates if Chem.MolFromSmiles(c.strip().replace(" ", "")) is not None][:3]

    # FORCE CLEANUP: Strip spaces and newlines to prevent RDKit crashes
    clean_smiles = []
    for s in smiles:
        if isinstance(s, str):
            clean_s = s.strip().replace(" ", "").replace("\n", "")
            clean_smiles.append(clean_s)

    write_report_event(state["thread_id"], f"Generated {len(clean_smiles)} candidates:\n" + "\n".join(clean_smiles), title=f"GENERATION ITERATION {iter_num}")

    return {
        "current_smiles": clean_smiles, 
        "iteration": iter_num,
        "eval_results": [], 
        "history_log": [f"[Iteration {iter_num}] Generated {len(clean_smiles)} SMILES."]
    }

def evaluate_candidates(state: DrugDesignState):
    """Calculates metrics and safe 3D Shape."""
    results = []
    for smiles in state["current_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                "smiles": smiles, "valid": False, "qed": 0.0, 
                "logp": 0.0, "mw": 0.0, "docking": 0.0, "total_score": -100.0
            })
            continue
            
        qed_val = round(float(QED.qed(mol)), 3)
        logp_val = round(float(Descriptors.MolLogP(mol)), 2)
        mw_val = round(float(Descriptors.MolWt(mol)), 2)
        
        shape_sim = 0.0
        try:
            mol_3d = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxAttempts = 1000
            
            if AllChem.EmbedMolecule(mol_3d, params) != -1:
                AllChem.MMFFOptimizeMolecule(mol_3d)
                shape_sim = rdShapeHelpers.ShapeTanimotoDist(ref_mol, mol_3d)
        except Exception:
            pass 

        docking_score = round(-10.0 * (1.0 - shape_sim), 2)
        total_score = abs(docking_score) * 2.0
        if mw_val > 1200 or mw_val < 500:
            total_score -= 20.0
            
        results.append({
            "smiles": smiles, "valid": True, "qed": qed_val, 
            "logp": logp_val, "mw": mw_val, "docking": docking_score, "total_score": total_score
        })
        
    return {"eval_results": results}

def aggregate(state: DrugDesignState):
    candidates = state["eval_results"]
    candidates.sort(key=lambda x: x.get("total_score", -100), reverse=True)
    best = candidates[0] if candidates else {}

    log_content = ""
    for cand in candidates:
        v = "✅" if cand.get('valid') else "❌"
        log_content += f"[{v}] SMILES: {cand.get('smiles', 'N/A')}\n    MW: {cand.get('mw')} | QED: {cand.get('qed')} | Docking: {cand.get('docking')} | Score: {cand.get('total_score')}\n\n"
        
    write_report_event(state["thread_id"], log_content.strip(), title=f"EVALUATION RESULTS ITERATION {state['iteration']}")
    return {"best_candidate": best}

def post_search(state: DrugDesignState):
    top = state.get("best_candidate", {})
    if not top: return {}
        
    top_smi = top.get("smiles", "")
    exists = False
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{top_smi}/cids/JSON"
        if requests.get(url, timeout=3).status_code == 200: exists = True
    except Exception: pass
        
    write_report_event(state["thread_id"], f"Top Candidate: {top_smi}\nExists in Literature: {exists}", title="NOVELTY CHECK")
    return {"history_log": [f"[Post-search] Checked PubChem presence: {exists}"]}

def critic(state: DrugDesignState):
    latest = state.get("best_candidate", {})
    if not latest or not latest.get("valid", False):
        fb = "Previous candidate was chemically invalid. Ensure valences are correct and rings are closed."
    else:
        prompt = f"Current Candidate: {latest['smiles']}\nMolecular Weight: {latest['mw']} Da\nQED Score: {latest['qed']}\nShape Score: {latest['docking']}\nGive a concise 1-2 sentence medicinal chemistry critique explaining how to optimize the linker length or attachment point to improve the 3D shape while keeping MW within 700-1100."
        try:
            fb = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            fb = "Adjust the linker length to improve 3D alignment."

    write_report_event(state["thread_id"], fb, title="MEDCHEM CRITIQUE")
    return {"feedback": fb, "history_log": [f"[Critic Feedback] {fb}"]}

def route_next_step(state: DrugDesignState) -> str:
    if state["iteration"] >= 3: return "finalize"
    top = state.get("best_candidate", {})
    if top.get("valid") and top.get("docking", 0) <= -6.0 and 500 <= top.get("mw", 2000) <= 1200:
        return "finalize"
    return "generate"

def finalize(state: DrugDesignState):
    final = state.get("best_candidate", {})
    write_report_event(state["thread_id"], f"Final Selected SMILES: {final.get('smiles')}\nStats: MW {final.get('mw')} | QED {final.get('qed')}", title="WORKFLOW COMPLETE")
    return {"history_log": ["[Finalize] Design process complete."]}

# GRAPH COMPILATION
builder = StateGraph(DrugDesignState)
builder.add_node("ingest", ingest)
builder.add_node("pre_search", pre_search_custom_node)
builder.add_node("generate", generate_custom_node)
builder.add_node("evaluate", evaluate_candidates)
builder.add_node("aggregate", aggregate)
builder.add_node("post_search", post_search)
builder.add_node("critic", critic)
builder.add_node("finalize", finalize)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "pre_search")
builder.add_edge("pre_search", "generate")
builder.add_edge("generate", "evaluate")
builder.add_edge("evaluate", "aggregate")
builder.add_edge("aggregate", "post_search")
builder.add_edge("post_search", "critic")
builder.add_conditional_edges("critic", route_next_step, {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

drug_agent = builder.compile(checkpointer=MemorySaver())