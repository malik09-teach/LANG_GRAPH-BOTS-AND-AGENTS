import streamlit as st
import requests
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(layout="wide", page_title="AI PROTAC Designer")

st.title("🧬 Autonomous PROTAC Designer")
st.markdown("Powered by LangGraph, FastAPI, PubChem REST, and RDKit 3D")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("Target Configuration")
    target_input = st.text_area(
        "Target Name or Constraints", 
        value="Epidermal Growth Factor Receptor (EGFR)",
        height=70
    )
    
    session_id = st.text_input("Run/Session ID (thread_id)", value="RUN_EGFR_001")
    
    if st.button("Initialize Agent", type="primary", use_container_width=True):
        with st.spinner("Agent is analyzing the target, auto-discovering inhibitors, and designing PROTACs..."):
            
            payload = {
                "target_info": target_input, 
                "thread_id": session_id
            }
            
            try:
                response = requests.post("http://localhost:8000/design-run", json=payload, timeout=180)
                if response.status_code == 200:
                    data = response.json()
                    best_mol = data.get("best_candidate", {})
                    
                    st.session_state["auto_pdb"] = data.get("auto_pdb_id")
                    st.session_state["auto_e3"] = data.get("auto_e3_ligase")
                    
                    if best_mol and best_mol.get("smiles"):
                        st.session_state["best_mol"] = best_mol
                        st.success(f"Agent converged! Report written to disk:\n`{data.get('report_file')}`")
                    else:
                        st.error("Agent finished, but no valid candidate was produced.")
                else:
                    st.error(f"Backend Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection failed. Is the FastAPI backend running? Error: {e}")

    # Display Auto-Extracted Parameters
    if "auto_pdb" in st.session_state:
        st.markdown("---")
        st.subheader("🤖 Agent Auto-Discovered Parameters")
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Discovered PDB ID", st.session_state["auto_pdb"])
        p_col2.metric("Assigned E3 Anchor", st.session_state["auto_e3"])

    st.markdown("---")
    st.subheader("📄 Live Report Log")
    if st.button("Read Saved txt File", use_container_width=True):
        try:
            res = requests.get(f"http://localhost:8000/report/{session_id}")
            if res.status_code == 200:
                st.code(res.json()["content"], language="text")
            else:
                st.warning("Report file not found on disk for this thread_id.")
        except Exception as e:
            st.error(f"Failed to fetch report: {e}")

with col2:
    st.subheader("3D Molecular Geometry")
    
    if "best_mol" in st.session_state and st.session_state["best_mol"]:
        best_mol_data = st.session_state["best_mol"]
        best_smiles = best_mol_data.get("smiles", "")
        
        st.write(f"**Top Candidate (SMILES):**\n`{best_smiles}`")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Shape Score", f"{best_mol_data.get('docking', 'N/A')}")
        m_col2.metric("QED Score", f"{best_mol_data.get('qed', 'N/A')}")
        m_col3.metric("LogP", f"{best_mol_data.get('logp', 'N/A')}")
        m_col4.metric("Weight", f"{best_mol_data.get('mw', 'N/A')} Da")
        
        with st.spinner("Rendering massive PROTAC structure in 3D..."):
            try:
                # 1. Clean spaces/newlines to prevent crash
                clean_smiles = best_smiles.strip().replace(" ", "").replace("\n", "")
                mol = Chem.MolFromSmiles(clean_smiles)
                
                # 2. Safety check in case the LLM hallucinated entirely invalid chemistry
                if mol is None:
                    st.error("RDKit Error: The AI generated a chemically invalid SMILES string. Please click 'Initialize Agent' to run another optimization cycle.")
                else:
                    mol = Chem.AddHs(mol)
                    
                    # 3. Robust parameters for embedding massive PROTACs
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    params.useRandomCoords = True
                    params.maxAttempts = 1000
                    
                    res = AllChem.EmbedMolecule(mol, params)
                    
                    if res == -1:
                        st.warning("RDKit could not resolve 3D coordinates for this massive PROTAC structure. (Common for MW > 1000).")
                    else:
                        AllChem.MMFFOptimizeMolecule(mol)
                        mol_block = Chem.MolToMolBlock(mol)
                        
                        viewer = py3Dmol.view(width=700, height=520)
                        viewer.addModel(mol_block, "mol")
                        viewer.setStyle({'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.2}})
                        viewer.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'color': 'white'})
                        viewer.setBackgroundColor('#0E1117')
                        viewer.zoomTo()
                        
                        showmol(viewer, height=520, width=700)
            except Exception as e:
                st.warning(f"Could not render 3D model for structure. Error: {e}")
    else:
        st.info("👈 Enter a protein target and click 'Initialize Agent' to generate and visualize a PROTAC.")