# app.py
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from model import GNN
import os
from torch_geometric.data import Data
import torch.nn.functional as F

@st.cache_data
def mol_to_graph_data_obj(mol_block: str):
    mol = Chem.MolFromMol2Block(mol_block, sanitize=True)

    # Build edge index and edge attributes
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bond.GetBondTypeAsDouble()]] * 2
    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    # Build node feature matrix
    atom_feats = []
    
    for atom in mol.GetAtoms():
        atom_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalValence(), 
            atom.GetTotalNumHs()
        ])
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Creating a single‑graph batch with batch index = 0
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)  # all nodes belong to graph 0

    return data,mol

model_path = "best_model.pt"
st.title("Protein-Ligand Binding Affinity Predictor")

@st.cache_resource
def load_model():
    model = GNN(in_c=4, hid=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# upload ligand
uploaded_file = st.file_uploader("Upload a ligand file (.mol2)", type=["mol2"])
if uploaded_file:
    mol_block = uploaded_file.read().decode("utf-8")
    result = mol_to_graph_data_obj(mol_block)

    if result is None:
        st.error("Could not parse molecule or featurize it.")
    else:
        data, mol = result
        st.subheader("Molecule Preview")
        st.image(Draw.MolToImage(mol), caption="Ligand Molecule")

        # Load model and predict
        if os.path.exists(model_path):
            model = load_model()
            with torch.no_grad():
                output = model(data)
                predicted_affinity = output.item()
            st.success(f"Predicted Binding Affinity (pKd/pKi): {predicted_affinity:.3f}")
        else:
            st.error("Model file not found.")