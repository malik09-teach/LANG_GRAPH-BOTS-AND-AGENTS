import operator
from typing import Annotated, List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ==========================================
# REAL CHEMISTRY LIBRARIES
# ==========================================
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from meeko import MoleculePreparation  # Official tool for RDKit -> Vina PDBQT
from vina import Vina                # AutoDock Vina API

class DrugDesignState(TypedDict):
    target_info: str
    current_smiles: List[str]
    eval_results: Annotated[List[Dict[str, Any]], operator.add] 
    best_candidates: Annotated[List[Dict[str, Any]], operator.add]
    iteration: int
    feedback: str

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def ingest_target(state: DrugDesignState):
    print(f"--- [Node: Ingest Target] ---")
    return {"iteration": 0}

def generate_molecules(state: DrugDesignState):
    print(f"--- [Node: Generate Molecules] Iteration {state['iteration']} ---")
    
    prompt = f"""
    You are an expert computational chemist. 
    Target Info: {state['target_info']}
    Previous Feedback: {state.get('feedback', 'No previous feedback.')}
    
    Output exactly ONE valid SMILES string representing a candidate molecule. 
    Do not output any markdown, explanations, or text other than the SMILES string.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    generated_smiles = response.content.strip()
    
    return {
        "current_smiles": [generated_smiles], 
        "iteration": state["iteration"] + 1,
        "eval_results": [] 
    }

# ==========================================
# ACTUAL TOOL GROUNDING NODES
# ==========================================

def eval_rdkit(state: DrugDesignState):
    """Calculates true chemical validity, QED, MW, and LogP."""
    print(f"--- [Node: Eval RDKit] ---")
    results = []
    
    for smiles in state["current_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        
        # Grounding: If the LLM hallucinated bad chemistry, penalize it instantly
        if mol is None:
            results.append({"smiles": smiles, "source": "RDKit", "valid": False, "qed": 0.0, "logp": 0.0, "mw": 0.0})
            continue
            
        # Grounding: Calculate exact properties
        qed_score = QED.qed(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        results.append({
            "smiles": smiles, 
            "source": "RDKit", 
            "valid": True, 
            "qed": round(qed_score, 3), 
            "logp": round(logp, 2), 
            "mw": round(mw, 2)
        })
        
    return {"eval_results": results}

def eval_docking(state: DrugDesignState):
    """Executes a real AutoDock Vina physics simulation."""
    print(f"--- [Node: Eval Docking] ---")
    results = []
    
    # Initialize Real AutoDock Vina
    v = Vina(sf_name='vina')
    
    try:
        # Load your actual target file and define the 3D binding pocket coordinates
        v.set_receptor('receptor.pdbqt')
        v.compute_vina_maps(center=[10.5, 15.2, 20.8], box_size=[20, 20, 20])
        receptor_loaded = True
    except Exception as e:
        print(f"WARNING: Could not load receptor.pdbqt. Docking will be skipped. Error: {e}")
        receptor_loaded = False

    for smiles in state["current_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not receptor_loaded:
            results.append({"smiles": smiles, "source": "Docking", "score": 0.0})
            continue
            
        try:
            # 1. Add Hydrogens and generate 3D geometry
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 2. Convert RDKit Molecule to Vina PDBQT format using Meeko
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()
            
            # 3. Run the Docking Simulation
            v.set_ligand_from_string(pdbqt_string)
            v.dock(exhaustiveness=8, n_poses=1)
            
            # 4. Extract real binding affinity (kcal/mol) - lower is better
            energy = v.score()[0] 
            results.append({"smiles": smiles, "source": "Docking", "score": round(energy, 2)})
            
        except Exception as e:
            print(f"Docking calculation failed for {smiles}: {e}")
            results.append({"smiles": smiles, "source": "Docking", "score": 0.0})
            
    return {"eval_results": results}

def aggregate_and_score(state: DrugDesignState):
    """Merges real RDKit and Docking data into a single candidate profile."""
    print(f"--- [Node: Aggregate] ---")
    smiles_data = {}
    
    for res in state["eval_results"]:
        s = res["smiles"]
        if s not in smiles_data:
            smiles_data[s] = {"smiles": s, "total_score": 0}
        
        if res["source"] == "RDKit":
            smiles_data[s]["valid"] = res["valid"]
            smiles_data[s]["qed"] = res["qed"]
            smiles_data[s]["logp"] = res["logp"]
            smiles_data[s]["mw"] = res["mw"]
            # Heavily penalize invalid structures
            if not res["valid"]:
                smiles_data[s]["total_score"] -= 100 
            else:
                smiles_data[s]["total_score"] += (res["qed"] * 10)
                
        elif res["source"] == "Docking":
            smiles_data[s]["docking"] = res["score"]
            # Good docking scores are negative, so subtract it to increase total_score
            smiles_data[s]["total_score"] -= res["score"] 
            
    best_cands = list(smiles_data.values())
    return {"best_candidates": best_cands}

def critic(state: DrugDesignState):
    """Feeds the real physics data back to the LLM to learn and adapt."""
    print(f"--- [Node: Critic] ---")
    
    latest_candidates = state.get("best_candidates", [])
    if not latest_candidates:
        return {"feedback": "No valid candidates generated. Ensure correct SMILES syntax."}
        
    top_candidate = latest_candidates[-1]
    
    if not top_candidate.get("valid", False):
        return {"feedback": f"The SMILES string '{top_candidate['smiles']}' was chemically invalid. Generate a valid structure."}
    
    prompt = f"""
    You generated: {top_candidate['smiles']}
    Actual Molecular Weight: {top_candidate.get('mw', 'N/A')} Da
    Actual LogP: {top_candidate.get('logp', 'N/A')}
    Actual QED Score: {top_candidate.get('qed', 'N/A')}
    Actual Docking Affinity: {top_candidate.get('docking', 'N/A')} kcal/mol
    
    Critique this molecule based on these real physical scores. Provide a 1-sentence instruction 
    on how to modify the functional groups to improve the binding affinity and QED in the next iteration.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"feedback": response.content}

def route_next_step(state: DrugDesignState) -> str:
    current_iter = state["iteration"]
    if current_iter >= 5:
        return "finalize"
    
    latest_candidates = state.get("best_candidates", [])
    if latest_candidates:
        top_cand = latest_candidates[-1]
        # Target condition: QED > 0.6 AND Docking Score < -8.0 kcal/mol
        if top_cand.get("qed", 0) > 0.6 and top_cand.get("docking", 0) < -8.0: 
            return "finalize"
            
    return "generate"

def finalize(state: DrugDesignState):
    print(f"--- [Node: Finalize] ---")
    return state

# ==========================================
# GRAPH COMPILATION
# ==========================================
builder = StateGraph(DrugDesignState)

builder.add_node("ingest", ingest_target)
builder.add_node("generate", generate_molecules)
builder.add_node("eval_rdkit", eval_rdkit)
builder.add_node("eval_docking", eval_docking)
builder.add_node("aggregate", aggregate_and_score)
builder.add_node("critic", critic)
builder.add_node("finalize", finalize)

builder.add_edge(START, "ingest")
builder.add_edge("ingest", "generate")
builder.add_edge("generate", "eval_rdkit")
builder.add_edge("generate", "eval_docking")
builder.add_edge("eval_rdkit", "aggregate")
builder.add_edge("eval_docking", "aggregate")
builder.add_edge("aggregate", "critic")
builder.add_conditional_edges("critic", route_next_step, {"generate": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

checkpointer = MemorySaver()
drug_agent = builder.compile(checkpointer=checkpointer)