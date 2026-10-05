import streamlit as st
import requests
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(layout="wide", page_title="AI PROTAC Designer")

st.title(" Autonomous Protein Inhibitor & PROTAC Designer")
st.markdown("Powered by LangGraph, FastAPI, PubChem REST, and RDKit 3D")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("Target Configuration")
    target_input = st.text_area(
        "Target Name or Constraints", 
        value="MYC Transcription Factor\nDesign Strategy: PROTAC Degradation",
        height=140
    )
    
    session_id = st.text_input("Run/Session ID (thread_id)", value="run_1001")
    
    if st.button("Initialize Agent", type="primary", use_container_width=True):
        with st.spinner("Discovering PDB ID, assigning E3 ligase, and generating 3D PROTACs..."):
            payload = {"target_info": target_input, "thread_id": session_id}
            
            try:
                response = requests.post("http://localhost:8000/design-run", json=payload, timeout=180)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    
                    st.session_state["auto_pdb"] = data.get("auto_pdb_id")
                    st.session_state["auto_e3"] = data.get("auto_e3_ligase")
                    
                    if candidates:
                        sorted_cands = sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)
                        st.session_state["best_mol"] = sorted_cands[0]
                        st.success(f"Agent converged! Report written to disk:\n`{data.get('report_file')}`")
                    else:
                        st.error("No valid candidate produced.")
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Display Auto-Extracted Parameters
    if "auto_pdb" in st.session_state:
        st.markdown("---")
        st.subheader(" Agent Auto-Discovered Parameters")
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Discovered PDB ID", st.session_state["auto_pdb"])
        p_col2.metric("Assigned E3 Anchor", st.session_state["auto_e3"])

    st.markdown("---")
    st.subheader(" Live Report Log")
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
        
        st.write(f"**Top Candidate (SMILES):** `{best_smiles}`")
        
        # Display Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Binding Affinity", f"{best_mol_data.get('docking', 'N/A')} kcal/mol")
        m_col2.metric("QED Score", f"{best_mol_data.get('qed', 'N/A')}")
        m_col3.metric("LogP", f"{best_mol_data.get('logp', 'N/A')}")
        m_col4.metric("Weight", f"{best_mol_data.get('mw', 'N/A')} Da")
        
        # 3D Visualization Render
        try:
            mol = Chem.MolFromSmiles(best_smiles)
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates using ETKDGv3
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            AllChem.EmbedMolecule(mol, params)
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
        st.info("Enter target configurations and click 'Initialize Agent' to generate and visualize.")