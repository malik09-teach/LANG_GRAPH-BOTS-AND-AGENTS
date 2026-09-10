import streamlit as st
import requests
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(layout="wide", page_title="AI Drug Designer")

st.title("🧬 Autonomous De Novo Drug Designer")
st.markdown("Powered by LangGraph, FastAPI, AutoDock Vina, and RDKit")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Target Configuration")
    target_input = st.text_area(
        "Disease & Target Constraints", 
        value="Target: MYC Transcription Factor...\nDesign Strategy: Molecular Glue...",
        height=200
    )
    
    session_id = st.text_input("Run/Session ID", value="run_001")
    
    if st.button("Initialize Agent", type="primary"):
        with st.spinner("Agent is running cycles (Generate → Dock → Critic). This may take a minute..."):
            
            payload = {"target_info": target_input, "thread_id": session_id}
            
            try:
                # Call the FastAPI backend
                response = requests.post("http://localhost:8000/design-run", json=payload, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    report_file = data.get("report_file", "report.txt")
                    
                    if candidates:
                        # Sort by score to get the best one
                        sorted_cands = sorted(candidates, key=lambda x: x.get("total_score", 0), reverse=True)
                        st.session_state["best_mol"] = sorted_cands[0]
                        st.success(f"Agent successfully converged! Full report saved to: `{report_file}`")
                    else:
                        st.error("Agent failed to find a valid candidate within iteration limits.")
                else:
                    st.error(f"Backend Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is FastAPI running on port 8000?")

with col2:
    st.subheader("3D Molecular Visualization")
    
    if "best_mol" in st.session_state:
        best_mol_data = st.session_state["best_mol"]
        best_smiles = best_mol_data.get("smiles", "")
        
        st.write(f"**Top Candidate (SMILES):** `{best_smiles}`")
        
        # Display the real physical metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Binding Affinity", f"{best_mol_data.get('docking', 'N/A')} kcal/mol")
        m_col2.metric("QED Score", f"{best_mol_data.get('qed', 'N/A')}")
        m_col3.metric("LogP", f"{best_mol_data.get('logp', 'N/A')}")
        m_col4.metric("Weight", f"{best_mol_data.get('mw', 'N/A')} Da")
        
        # Render the 3D Model
        try:
            mol = Chem.MolFromSmiles(best_smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            mol_block = Chem.MolToMolBlock(mol)
            
            viewer = py3Dmol.view(width=600, height=500)
            viewer.addModel(mol_block, "mol")
            viewer.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
            viewer.setBackgroundColor('#f5f5f5')
            viewer.zoomTo()
            
            showmol(viewer, height=500, width=600)
        except Exception as e:
            st.warning(f"Could not render 3D model for this structure. Ensure valid SMILES syntax. Error: {e}")
            
    else:
        st.info("Enter target configurations and click 'Initialize Agent' to generate and visualize.")