import streamlit as st
import requests
import io
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

st.set_page_config(layout="wide", page_title="AI PROTAC Designer")

st.title("⚡ Autonomous Protein Inhibitor & PROTAC Designer")
st.markdown("Powered by LangGraph, FastAPI, PubChem REST, and RDKit 3D")

col1, col2 = st.columns([1, 1.5], gap="large")

# ────────────────────────────────────────────────────────
# LEFT COLUMN — Configuration & Report
# ────────────────────────────────────────────────────────
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
                response = requests.post("http://localhost:8000/design-run", json=payload, timeout=300)
                if response.status_code == 200:
                    data = response.json()

                    # Use correct key "final_candidates" (matches backend)
                    candidates = data.get("final_candidates", [])
                    best_from_backend = data.get("best_candidate", {})

                    st.session_state["auto_pdb"] = data.get("auto_pdb_id")
                    st.session_state["auto_e3"] = data.get("auto_e3_ligase")
                    st.session_state["target_name"] = data.get("target_name", "")
                    st.session_state["report_file"] = data.get("report_file", "")

                    # Store ALL candidates so every PROTAC molecule is viewable
                    if candidates:
                        sorted_cands = sorted(
                            candidates,
                            key=lambda x: x.get("total_score", -999),
                            reverse=True,
                        )
                        st.session_state["all_candidates"] = sorted_cands
                        st.session_state["best_mol"] = sorted_cands[0]
                        st.success(
                            f"✅ Agent converged with **{len(sorted_cands)}** candidates!\n\n"
                            f"Report saved to: `{data.get('report_file')}`"
                        )
                    elif best_from_backend:
                        st.session_state["all_candidates"] = [best_from_backend]
                        st.session_state["best_mol"] = best_from_backend
                        st.success("Agent converged with 1 candidate (best only).")
                    else:
                        st.error("No valid candidate produced by the agent.")
                else:
                    st.error(f"Backend Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot reach backend at http://localhost:8000.\nStart it with: `python backend.py`")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Display Auto-Extracted Parameters
    if "auto_pdb" in st.session_state:
        st.markdown("---")
        st.subheader("🤖 Agent Auto-Discovered Parameters")
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Discovered PDB ID", st.session_state.get("auto_pdb", "—"))
        p_col2.metric("Assigned E3 Anchor", st.session_state.get("auto_e3", "—"))

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


# ────────────────────────────────────────────────────────
# HELPER: Render a 2D structure image from SMILES
# ────────────────────────────────────────────────────────
def render_2d_image(smiles: str):
    """Generate a 2D depiction of the molecule and display it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning(f"RDKit could not parse SMILES for 2D drawing: `{smiles}`")
        return

    # Compute 2D coordinates for a clean layout
    AllChem.Compute2DCoords(mol)

    # Draw to PIL Image
    img = Draw.MolToImage(mol, size=(700, 400), kekulize=True)

    # Convert PIL Image → bytes for st.image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, caption="2D Structure", use_container_width=True)


# ────────────────────────────────────────────────────────
# HELPER: Render a 3D viewer for a SMILES string
# ────────────────────────────────────────────────────────
def render_3d_molecule(smiles: str, width: int = 680, height: int = 450):
    """Generate a 3D conformer from SMILES and render with py3Dmol."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning(f"RDKit could not parse SMILES for 3D: `{smiles}`")
        return False

    mol = Chem.AddHs(mol)

    # ── Attempt ETKDGv3 first (best quality) ──
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    params.maxAttempts = 5000       # more attempts for large PROTACs
    params.numThreads = 0           # use all available threads

    embed_result = AllChem.EmbedMolecule(mol, params)

    # ── Fallback: plain random-coords embedding ──
    if embed_result == -1:
        fallback_params = AllChem.EmbedParameters()
        fallback_params.useRandomCoords = True
        fallback_params.maxAttempts = 10000
        fallback_params.randomSeed = 123
        embed_result = AllChem.EmbedMolecule(mol, fallback_params)

    if embed_result == -1:
        st.warning("3D embedding failed — molecule may be too constrained. Showing 2D only.")
        return False

    # ── Optimize geometry: try MMFF first, fall back to UFF ──
    try:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if mmff_result == -1:
            raise RuntimeError("MMFF failed")
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # use un-optimized coords — still viewable

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {"colorscheme": "cyanCarbon", "radius": 0.2}})
    viewer.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "white"})
    viewer.setBackgroundColor("#0E1117")
    viewer.zoomTo()
    showmol(viewer, height=height, width=width)
    return True


# ────────────────────────────────────────────────────────
# RIGHT COLUMN — Per-Molecule 2D + 3D Viewer & Metrics
# ────────────────────────────────────────────────────────
with col2:
    st.subheader("Molecular Visualization — All PROTAC Candidates")

    all_cands = st.session_state.get("all_candidates", [])

    if all_cands:
        st.markdown(f"**{len(all_cands)} candidate(s)** ranked by total score:")

        for idx, cand in enumerate(all_cands):
            valid_icon = "✅" if cand.get("valid") else "❌"
            smiles = cand.get("smiles", "N/A")
            label = f"#{idx + 1}  {valid_icon}  Score {cand.get('total_score', '—')}  |  MW {cand.get('mw', '—')} Da"

            with st.expander(label, expanded=(idx == 0)):
                st.code(smiles, language="text")

                # Metrics row
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Binding Affinity", f"{cand.get('docking', 'N/A')} kcal/mol")
                mc2.metric("QED Score", f"{cand.get('qed', 'N/A')}")
                mc3.metric("LogP", f"{cand.get('logp', 'N/A')}")
                mc4.metric("Weight", f"{cand.get('mw', 'N/A')} Da")

                if cand.get("valid") and smiles != "N/A":
                    # ── 2D Structure Image ──
                    st.markdown("**🖼️ 2D Structure**")
                    try:
                        render_2d_image(smiles)
                    except Exception as e:
                        st.warning(f"Could not render 2D image. Error: {e}")

                    # ── 3D Interactive Viewer ──
                    st.markdown("**🧬 3D Conformer**")
                    try:
                        success = render_3d_molecule(smiles)
                        if not success:
                            st.info("Falling back to 2D view above.")
                    except Exception as e:
                        st.warning(f"Could not render 3D model. Error: {e}")
                else:
                    st.info("Molecule marked invalid — visualization unavailable.")
    else:
        st.info("Enter target configurations and click **Initialize Agent** to generate and visualize.")