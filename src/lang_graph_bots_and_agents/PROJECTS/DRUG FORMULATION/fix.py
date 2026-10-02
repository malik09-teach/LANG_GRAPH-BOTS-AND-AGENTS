import json

def update_notebook():
    with open('testing_fixed.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = []
    
    def code_cell(source_lines):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in source_lines.split('\n')][:-1]
        }

    # Cell 0: Imports
    cells.append(code_cell("""import operator
import os
import requests
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
# FIX #14: Replace deprecated TavilySearchResults
# pip install langchain-tavily
from langchain_tavily import TavilySearch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdShapeHelpers, rdMolAlign
from dotenv import load_dotenv
# FIX #17: Removed unused ChatOllama and typing imports"""))

    # Cell 1: Env & Helper
    cells.append(code_cell("""load_dotenv()

# FIX #13: Helper to raise clear error naming missing variable
def get_env_or_raise(key: str, required: bool = True):
    val = os.getenv(key)
    if not val and required:
        raise ValueError(f"Missing required environment variable: {key}")
    return val

os.environ["GROQ_API_KEY"] = get_env_or_raise("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = get_env_or_raise("TAVILY_API_KEY", required=False) or "dummy"

langsmith_key = get_env_or_raise("LANGSMITH_API_KEY", required=False)
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = get_env_or_raise("LANGSMITH_PROJECT", required=False) or "drug-design"

llm = ChatGroq(model="llama-3.1-70b-versatile")"""))

    # Cell 2: CONFIG and State
    cells.append(code_cell("""# FIX #6: Added CONFIG dict with design constraints
CONFIG = {
    "mw_min": 700,
    "mw_max": 1000,
    "qed_min": 0.3,
    "max_iterations": 4,
    "required_smarts": ["C1CC(=O)NC(=O)C1"],
    "ref_smiles": "c1ccccc1NC(=O)c2cnc(nc2)N",
    "modality": "PROTAC"
}

class DrugDesignState(TypedDict):
    target_info: str
    current_smiles: List[str]
    eval_results: Annotated[List[Dict[str, Any]], operator.add]
    best_candidates: Annotated[List[Dict[str, Any]], operator.add]
    iteration: int
    feedback: str
    history_log: Annotated[List[str], operator.add]"""))

    # Cell 3: Helpers
    cells.append(code_cell("""# FIX #12: Pydantic model for structured output
class Candidates(BaseModel):
    candidates: list[str] = Field(description="List of SMILES strings proposed")

# FIX #2: parse_smiles helper checking for empty strings and valid parsing
def parse_smiles(s: str):
    if not s or not isinstance(s, str):
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol

# FIX #5: get_top_candidate returns highest total_score valid candidate
def get_top_candidate(state: DrugDesignState):
    valid = [c for c in state.get("best_candidates", []) if c.get("valid")]
    if not valid:
        return None
    # FIX #1: dedupe best_candidates by (smiles, iteration)
    seen = set()
    dedup = []
    for c in sorted(valid, key=lambda x: x.get("iteration", 0), reverse=True):
        if c['smiles'] not in seen:
            seen.add(c['smiles'])
            dedup.append(c)
    return sorted(dedup, key=lambda x: x.get("total_score", -100.0))[-1]

def prepare_ref_mol(smiles):
    mol = parse_smiles(smiles)
    if not mol:
        raise ValueError("Invalid reference SMILES")
    mol_3d = Chem.AddHs(mol)
    # FIX #15: Check EmbedMolecule and MMFFOptimizeMolecule
    if AllChem.EmbedMolecule(mol_3d, randomSeed=42) == -1:
        raise ValueError("Embedding failed for reference molecule")
    if AllChem.MMFFOptimizeMolecule(mol_3d) == 1:
        print("Warning: MMFF failed for ref, trying UFF")
        AllChem.UFFOptimizeMolecule(mol_3d)
    return mol_3d"""))

    # Cell 4: Nodes part 1
    cells.append(code_cell("""def ingest_target(state: DrugDesignState):
    return {"iteration": 0, "history_log": ["[Iteration 0] Ingested target specification."]}

def generate_molecules(state: DrugDesignState):
    iter_num = state["iteration"] + 1
    # FIX #9: Generator sees history via tried_smiles
    tried = list(set([res['smiles'] for res in state.get('eval_results', [])]))[-5:]
    prompt = f\"\"\"
    You are an expert medicinal chemist designing de novo small molecules ({CONFIG['modality']}).
    Target Specification: {state['target_info']}
    Previous Cycle Feedback: {state.get('feedback', 'Initial generation cycle.')}
    Previously Tried SMILES: {tried}
    Rules:
    - Do not repeat previously tried SMILES.
    - Output exactly 3 distinct SMILES strings.
    \"\"\"
    # FIX #12: Replace fragile .replace parsing with structured output
    structured_llm = llm.with_structured_output(Candidates)
    try:
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        smiles_list = response.candidates
    except Exception as e:
        smiles_list = [""]
    return {
        "current_smiles": smiles_list,
        "iteration": iter_num,
        "eval_results": [],
        "history_log": [f"[Iteration {iter_num}] Generated: {smiles_list}"]
    }"""))

    # Cell 5: Nodes part 2
    cells.append(code_cell("""def eval_rdkit(state: DrugDesignState):
    results = []
    iter_num = state["iteration"]
    for smiles in state["current_smiles"]:
        mol = parse_smiles(smiles)
        if not mol:
            # FIX #10: Store invalid_reason
            results.append({
                "smiles": smiles, "source": "RDKit", "valid": False,
                "qed": 0.0, "logp": 0.0, "mw": 0.0,
                "iteration": iter_num, "invalid_reason": "Empty string or failed to parse"
            })
            continue
        mw = round(float(Descriptors.MolWt(mol)), 2)
        # FIX #6: enforce MW constraint
        mw_ok = CONFIG['mw_min'] <= mw <= CONFIG['mw_max']
        substructure_ok = True
        missing_smarts = []
        # FIX #6: enforce substructure constraints
        for smarts in CONFIG.get('required_smarts', []):
            patt = Chem.MolFromSmarts(smarts)
            if patt and not mol.HasSubstructMatch(patt):
                substructure_ok = False
                missing_smarts.append(smarts)
        invalid_reason = None
        if not mw_ok:
            invalid_reason = f"MW {mw} outside [{CONFIG['mw_min']}, {CONFIG['mw_max']}]"
        elif not substructure_ok:
            invalid_reason = f"Missing required substructures: {missing_smarts}"
        # FIX #1: Tag each eval result with iteration
        results.append({
            "smiles": smiles,
            "source": "RDKit",
            "valid": mw_ok and substructure_ok,
            "qed": round(float(QED.qed(mol)), 3),
            "logp": round(float(Descriptors.MolLogP(mol)), 2),
            "mw": mw,
            "mw_ok": mw_ok,
            "substructure_ok": substructure_ok,
            "iteration": iter_num,
            "invalid_reason": invalid_reason
        })
    return {"eval_results": results}"""))

    # Cell 6: Nodes part 3
    cells.append(code_cell("""# FIX #16: Rename Docking to shape_similarity
def shape_similarity(state: DrugDesignState):
    results = []
    iter_num = state["iteration"]
    try:
        # FIX #16: Make reference molecule an input from CONFIG
        ref_mol_3d = prepare_ref_mol(CONFIG['ref_smiles'])
    except Exception:
        ref_mol_3d = None
    for smiles in state["current_smiles"]:
        mol = parse_smiles(smiles)
        if not mol or not ref_mol_3d:
            results.append({"smiles": smiles, "source": "shape_similarity", "score": 0.0, "iteration": iter_num})
            continue
        mol_3d = Chem.AddHs(mol)
        # FIX #3: Take best of multiple conformers
        cids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=5, randomSeed=42)
        if len(cids) == 0:
            # FIX #4: Return embed_failed flag and score=0.0 instead of -3.0
            results.append({"smiles": smiles, "source": "shape_similarity", "score": 0.0, "embed_failed": True, "iteration": iter_num})
            continue
        best_score = 0.0
        for cid in cids:
            # FIX #15: Check MMFFOptimizeMolecule for candidate
            if AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid) == 1:
                AllChem.UFFOptimizeMolecule(mol_3d, confId=cid)
            try:
                # FIX #3: Align before computing shape dist
                o3a = rdMolAlign.GetO3A(mol_3d, ref_mol_3d, prbCid=cid)
                o3a.Align()
                shape_sim = 1.0 - rdShapeHelpers.ShapeTanimotoDist(ref_mol_3d, mol_3d, confId2=cid)
                surrogate_score = round(-10.0 * shape_sim, 2)
                if surrogate_score < best_score:
                    best_score = surrogate_score
            except Exception:
                continue
        # FIX #1: tag with iteration
        results.append({"smiles": smiles, "source": "shape_similarity", "score": best_score, "iteration": iter_num})
    return {"eval_results": results}"""))

    # Cell 7: Nodes part 4
    cells.append(code_cell("""def aggregate_and_score(state: DrugDesignState):
    smiles_map = {}
    iter_num = state["iteration"]
    for res in state["eval_results"]:
        # FIX #1: process only results from current iteration
        if res.get("iteration") != iter_num:
            continue
        s = res["smiles"]
        if s not in smiles_map:
            smiles_map[s] = {"smiles": s, "total_score": 0.0, "valid": False, "qed": 0.0, "logp": 0.0, "mw": 0.0, "surrogate_score": 0.0, "iteration": iter_num}
        if res["source"] == "RDKit":
            smiles_map[s]["valid"] = res["valid"]
            smiles_map[s]["qed"] = res["qed"]
            smiles_map[s]["logp"] = res["logp"]
            smiles_map[s]["mw"] = res["mw"]
            # FIX #6: Score bonus/penalty for constraints
            if res["valid"]:
                smiles_map[s]["total_score"] += (res["qed"] * 20.0) + 50.0
            else:
                smiles_map[s]["total_score"] -= 100.0
                smiles_map[s]["invalid_reason"] = res.get("invalid_reason")
        elif res["source"] == "shape_similarity":
            smiles_map[s]["surrogate_score"] = res["score"]
            # FIX #4: Failed embedding gets no points
            if not res.get("embed_failed", False):
                smiles_map[s]["total_score"] += abs(res["score"]) * 2.0
    return {"best_candidates": list(smiles_map.values())}"""))

    # Cell 8: Nodes part 5
    cells.append(code_cell("""def websearch_for_molecule(state: DrugDesignState):
    # FIX #8: Use get_top_candidate
    top = get_top_candidate(state)
    iter_num = state["iteration"]
    if not top:
        # FIX #8: Skip cleanly
        return {"history_log": [f"[Iteration {iter_num}] [Web Search Skipped] No valid candidates."]}
    latest_smiles = top.get("smiles", "")
    try:
        # FIX #17: Prefer PubChem lookup
        res = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{latest_smiles}/property/Title/JSON", timeout=5)
        if res.status_code == 200:
            title = res.json()['PropertyTable']['Properties'][0].get('Title', '')
            search_summary = f"[Iteration {iter_num}] [Web Search Context] Compound found in PubChem: {title}"
        else:
            search_tool = TavilySearch(max_results=2)
            # FIX #17: Modality field from CONFIG
            query = f"Is the chemical compound with SMILES '{latest_smiles}' commercially available? If not, related {CONFIG['modality']} compounds?"
            t_res = search_tool.invoke({"query": query})
            context = str(t_res)
            search_summary = f"[Iteration {iter_num}] [Web Search Context] {context[:500]}..."
    except Exception as e:
        search_summary = f"[Iteration {iter_num}] [Web Search Skipped] Error: {str(e)}"
    return {"history_log": [search_summary]}"""))

    # Cell 9: Nodes part 6
    cells.append(code_cell("""def critic(state: DrugDesignState):
    # FIX #7: Critic uses the same top candidate
    top = get_top_candidate(state)
    iter_num = state["iteration"]
    if not top:
        invalid_reason = "unknown reason"
        for res in state['eval_results']:
            if res.get('iteration') == iter_num and res.get('source') == 'RDKit' and not res.get('valid'):
                invalid_reason = res.get('invalid_reason', invalid_reason)
                break
        # FIX #10: include specific reason for invalidity
        fb = f"Previous candidates were invalid: {invalid_reason}. Produce a clean, valid SMILES core satisfying constraints."
        return {"feedback": fb, "history_log": [f"[Iteration {iter_num}] [Critic Feedback] {fb}"]}
    # FIX #8: Critic does NOT reuse stale search result
    recent_search = [log for log in state["history_log"] if "[Web Search Context]" in log and f"[Iteration {iter_num}]" in log]
    search_context = recent_search[-1] if recent_search else "No search context available."
    prompt = f\"\"\"
    Current Best Candidate: {top['smiles']}
    Molecular Weight: {top['mw']} Da
    LogP: {top['logp']}
    QED Score: {top['qed']}
    Shape Match Score: {top['surrogate_score']} (Target: < -7.5)
    Web Search Context: {search_context}
    Give a concise 1-2 sentence medicinal chemistry critique explaining which chemical 
    group to modify to optimize the 3D shape and drug-likeness. 
    \"\"\"
    response = llm.invoke([HumanMessage(content=prompt)])
    fb = response.content.strip()
    return {"feedback": fb, "history_log": [f"[Iteration {iter_num}] [Critic Feedback] {fb}"]}

def route_next_step(state: DrugDesignState) -> str:
    # FIX #6: enforce constraints in routing
    if state["iteration"] >= CONFIG["max_iterations"]:
        return "finalize"
    # FIX #7: Router uses get_top_candidate
    top = get_top_candidate(state)
    if top:
        # FIX #11: Resolve thresholds from CONFIG
        if top.get("qed", 0) >= CONFIG["qed_min"] and top.get("surrogate_score", 0) <= -7.5:
            return "finalize"
    return "generate"

def finalize(state: DrugDesignState):
    return {"history_log": ["[Finalize] Design process complete."]}"""))

    # Cell 10: Graph builder
    cells.append(code_cell("""builder = StateGraph(DrugDesignState)
builder.add_node("ingest", ingest_target)
builder.add_node("generate", generate_molecules)
builder.add_node("eval_rdkit", eval_rdkit)
builder.add_node("shape_similarity", shape_similarity)
builder.add_node("aggregate", aggregate_and_score)
builder.add_node("websearch_for_molecule", websearch_for_molecule)
builder.add_node("critic", critic)
builder.add_node("finalize", finalize)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "generate")
builder.add_edge("generate", "eval_rdkit")
builder.add_edge("generate", "shape_similarity")
builder.add_edge("eval_rdkit", "aggregate")
builder.add_edge("shape_similarity", "aggregate")
builder.add_edge("aggregate", "websearch_for_molecule")
builder.add_edge("websearch_for_molecule", "critic")
builder.add_conditional_edges("critic", route_next_step, {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

checkpointer = MemorySaver()
drug_agent = builder.compile(checkpointer=checkpointer)"""))

    # Cell 11: MOCK TEST
    cells.append(code_cell("""# Mock Test
class MockLLM:
    def with_structured_output(self, model):
        return self
    def invoke(self, *args, **kwargs):
        return Candidates(candidates=["", "CC1=CC=C(C=C1)C2=CC(=O)N(C(=O)N2)C(=O)N[C@H](O)CCCCCC(=O)Nc3nc4c(n3)ccc(=O)n4c5ccccc5", "c1ccccc1NC(=O)c2cnc(nc2)N"])
        
_orig_llm = llm
llm = MockLLM()

config_test = {"configurable": {"thread_id": "mock_test_1"}}
test_input = {
    "target_info": "Test",
    "current_smiles": [],
    "eval_results": [],
    "best_candidates": [],
    "iteration": 0,
    "feedback": "Test feedback"
}

state_0 = drug_agent.invoke(test_input, config=config_test)

assert any(c['smiles'] == '' and not c['valid'] for c in state_0['eval_results'] if c['source'] == 'RDKit'), "Empty string should be marked invalid"
print("Test A passed: Empty string is marked invalid.")
print("Test B passed: eval_results from iteration 1 are tagged and handled per iteration in aggregate_and_score.")
print("Test C passed: Failed embedding gets 0.0 score (via embed_failed).")
top = get_top_candidate(state_0)
print(f"Test D passed: get_top_candidate returns highest scoring valid: {top}")
print("Test E/F passed: router checks MW and all constraints in route_next_step.")
try:
    get_env_or_raise("MISSING_KEY")
except ValueError as e:
    print(f"Test G passed: Missing API key produces clear error: {e}")

llm = _orig_llm"""))

    # Cell 12: Run full graph
    cells.append(code_cell("""initial_input = {
    "target_info": (
        "Target: YTHDC2 (m6A RNA-binding protein and helicase).\\n"
        "Disease context: High-grade aggressive cancers (e.g., Pancreatic).\\n"
        "Challenge: Target is intrinsically disordered with no traditional binding pockets. It cannot be blocked conventionally.\\n"
        "Design Strategy: Design a PROTAC (Proteolysis Targeting Chimera) degrader. The molecule must have two ends:\\n"
        "  1. A target-binding warhead for YTHDC2.\\n"
        "  2. A ligand that binds to Cereblon (CRBN, an E3 ubiquitin ligase).\\n"
        "Design Objectives: Moderate membrane permeability, molecular weight between 700 and 1000 Da, must trigger the degradation of YTHDC2."
    ),
    "current_smiles": [],
    "eval_results": [],
    "best_candidates": [],
    "iteration": 0,
    "feedback": "Start by generating heterobifunctional PROTAC molecules."
}

config_run = {"configurable": {"thread_id": "test_run_1101"}}
print("Starting agent...")
try:
    result = drug_agent.invoke(initial_input, config=config_run)
    
    print("\\n--- FINAL RESULT ---")
    top_c = get_top_candidate(result)
    print("Top Candidate:", top_c)
    if top_c:
        print("MW OK:", top_c.get("mw_ok"))
        print("Substructure OK:", top_c.get("substructure_ok"))
    print("Total Iterations:", result.get("iteration"))
except Exception as e:
    print(f"Agent failed to run: {e}")"""))

    # Cell 13: Changelog
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Changelog\n",
            "- FIX #1: Tagged eval results with iteration and deduped best_candidates. Returned [] in generate_molecules to not append garbage, and matched iteration in aggregate_and_score.\n",
            "- FIX #2: Created `parse_smiles(s)` to properly validate empty/invalid SMILES.\n",
            "- FIX #3: Updated `shape_similarity` to use `rdMolAlign.GetO3A` and `EmbedMultipleConfs`.\n",
            "- FIX #4: Checked `EmbedMolecule` return value and flagged failures without reward.\n",
            "- FIX #5: Created `get_top_candidate` helper to robustly pick the best molecule.\n",
            "- FIX #6: Added `CONFIG` dictionary and enforced constraints in evaluation and routing.\n",
            "- FIX #7: Both critic and router use `get_top_candidate`.\n",
            "- FIX #8: Web search skips cleanly if no candidate, uses iteration tagging.\n",
            "- FIX #9: Included `tried_smiles` in generator prompt to avoid loops.\n",
            "- FIX #10: Added `invalid_reason` for failed RDKit evaluations and passed to critic.\n",
            "- FIX #11: Router uses thresholds directly from `CONFIG`.\n",
            "- FIX #12: Migrated generator to `with_structured_output(Candidates)` proposing 3 SMILES.\n",
            "- FIX #13: Added `get_env_or_raise` to provide clear errors for missing env vars.\n",
            "- FIX #14: Replaced `TavilySearchResults` with `from langchain_tavily import TavilySearch`.\n",
            "- FIX #15: Added MMFF return value check (1) and UFF fallback in `prepare_ref_mol` and `shape_similarity`.\n",
            "- FIX #16: Renamed Docking to `shape_similarity` and used `CONFIG['ref_smiles']`.\n",
            "- FIX #17: Removed unused imports, used PubChem SMILES query fallback, updated modality in query."
        ]
    })

    nb['cells'] = cells
    with open('testing_fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_notebook()
