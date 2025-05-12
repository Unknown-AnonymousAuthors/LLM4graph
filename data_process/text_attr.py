from rdkit import Chem
from rdkit.Chem import Descriptors

import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Global variables to cache tokenizer and model, avoiding repeated loading
def init_chemberta(device='cuda' if torch.cuda.is_available() else 'cpu'):
    global_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
    global_model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM").to(device)
    print(f"ChemBERT running on: {device}")
    
    return global_tokenizer, global_model

def node_to_text(atom, mol):
    """Convert an RDKit atom to a natural language description with rich semantic information."""
    # Basic atom properties
    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    num_hydrogens = atom.GetTotalNumHs()
    hybridization = str(atom.GetHybridization()).lower()
    is_in_ring = atom.IsInRing()
    
    # Neighbor information
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
    neighbor_text = " and ".join(neighbors) if neighbors else "no atoms"
    
    # Functional group detection (simplified)
    functional_group = "none"
    if symbol == "O" and num_hydrogens > 0:
        functional_group = "hydroxyl group"
    elif symbol == "O" and degree == 2 and num_hydrogens == 0:
        functional_group = "carbonyl group"
    elif symbol == "N" and num_hydrogens > 0:
        functional_group = "amine group"
    elif symbol == "C" and hybridization == "sp2" and any(n.GetSymbol() == "O" for n in atom.GetNeighbors()):
        functional_group = "possible carboxyl or ester group"
    
    # Global molecular context
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings() if atom.GetIdx() in r]
    ring_text = f"in a {ring_sizes[0]}-membered ring" if ring_sizes else "not part of any ring"
    
    # Natural language description
    description = f"This is a {symbol} atom with a degree of {degree}, bonded to {num_hydrogens} hydrogen atoms. "
    description += f"It is {hybridization} hybridized and is {ring_text}. "
    description += f"The atom connects to {neighbor_text} and may be part of a {functional_group}. "
    description += f"It belongs to a molecule with a molecular weight of {mol_weight:.2f} Da and a logP of {logp:.2f}."
    
    return description

def bond_to_text(bond, mol):
    """Convert an RDKit bond to a natural language description with rich semantic information."""
    # Basic bond properties
    bond_type = str(bond.GetBondType()).lower()
    stereo = str(bond.GetStereo()).replace("STEREO", "").lower() or "no"
    is_conjugated = bond.GetIsConjugated()
    
    # Connected atoms
    atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
    atom1_symbol = atom1.GetSymbol()
    atom2_symbol = atom2.GetSymbol()
    
    # Ring context
    ring_info = mol.GetRingInfo()
    is_in_ring = bond.IsInRing()
    ring_sizes = [len(r) for r in ring_info.BondRings() if bond.GetIdx() in r]
    ring_text = f"part of a {ring_sizes[0]}-membered ring" if ring_sizes else "not in any ring"
    
    # Global molecular context
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # Natural language description
    description = f"This is a {bond_type} bond connecting a {atom1_symbol} atom to a {atom2_symbol} atom. "
    description += f"It has {stereo} stereochemistry and is {'conjugated' if is_conjugated else 'not conjugated'}. "
    description += f"The bond is {ring_text}. "
    description += f"It is part of a molecule with a molecular weight of {mol_weight:.2f} Da and a logP of {logp:.2f}."
    
    return description

def get_chemberta_embeddings(texts, global_tokenizer, global_model, batch_size=32, n_components=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate ChemBERTa embeddings for a list of texts with PCA reduction."""
    # Ensure model and tokenizer are loaded
    tokenizer = global_tokenizer
    model = global_model
    
    # Tokenize texts in batches
    tokenized_inputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokenized_inputs.append(inputs)
    
    # Run model inference
    embeddings = []
    for inputs in tokenized_inputs:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Move output to CPU and convert to NumPy
        batch_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)  # (num_texts, 384)
    
    # Apply PCA reduction
    n_samples = embeddings.shape[0]
    n_components = min(n_components, n_samples, 8)
    if n_samples > 1:
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings[:, :n_components]
    
    return reduced_embeddings

def get_all_textual(smiles_list, global_tokenizer, global_model):
    node_texts_list = []
    bond_texts_list = []
    node_embeddings_list = []
    bond_embeddings_list = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        node_texts = [node_to_text(atom, mol) for atom in mol.GetAtoms()]
        bond_texts = [bond_to_text(bond, mol) for bond in mol.GetBonds()]
        node_texts_list.append(node_texts)
        bond_texts_list.append(bond_texts)
    
        # total_node_texts = [text for node_texts in node_texts_list for text in node_texts]
        # total_bond_texts = [text for bond_texts in bond_texts_list for text in bond_texts]
        
        node_embeddings = get_chemberta_embeddings(node_texts, global_tokenizer, global_model, n_components=8, batch_size=32)
        bond_embeddings = get_chemberta_embeddings(bond_texts, global_tokenizer, global_model, n_components=8, batch_size=32)
        node_embeddings_list.append(node_embeddings)
        bond_embeddings_list.append(bond_embeddings)
    
    return node_embeddings_list, bond_embeddings_list

def main(dataset_split_name):
    assert dataset_split_name in ['train', 'test', 'valid']
    global_tokenizer, global_model = init_chemberta()
    
    root = './data/finetune/qm9/homo/'
    smiles_list = []
    with open(root + f"{dataset_split_name}.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines][1:]
        
        for line in lines:
            _, smiles, _ = line.split('\t')
            smiles_list.append(smiles)
    
    node_embeddings_list, bond_embeddings_list = get_all_textual(smiles_list, global_tokenizer, global_model)
    print(len(node_embeddings_list), len(bond_embeddings_list), node_embeddings_list[0].shape, bond_embeddings_list[0].shape)
    
    torch.save((node_embeddings_list, bond_embeddings_list), root + f"{dataset_split_name}_textual.pt")

if __name__ == '__main__':
    main('train')
    main('test')
    main('valid')
    # # Check CUDA availability
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Falling back to CPU.")
    
    # # Example SMILES
    # smiles = '[H]N1[C@@]2([H])[C@@]([H])(C([H])([H])OC([H])([H])[H])OC([H])([H])[C@@]12[H]'
    
    # # Repeat for 100 molecules
    # smiles_list = [smiles] * 100
    # node_texts_list = []
    # bond_texts_list = []
    
    # for smi in smiles_list:
    #     mol = Chem.MolFromSmiles(smi)
    #     node_texts = [node_to_text(atom, mol) for atom in mol.GetAtoms()]
    #     bond_texts = [bond_to_text(bond, mol) for bond in mol.GetBonds()]
    #     node_texts_list.append(node_texts)
    #     bond_texts_list.append(bond_texts)
    
    # print('zz', node_texts_list[0])
    # # Aggregate texts
    # total_node_texts = [text for node_texts in node_texts_list for text in node_texts]  # 100 * 9 = 900
    # total_bond_texts = [text for bond_texts in bond_texts_list for text in bond_texts]  # 100 * 5 = 500
    
    # print(f"Total node texts: {len(total_node_texts)}")
    # print(f"Total bond texts: {len(total_bond_texts)}")
    # print(f"Sample node text: {total_node_texts[0]}")
    # print(f"Sample bond text: {total_bond_texts[0]}")
    
    # # Generate embeddings
    # node_embeddings = get_chemberta_embeddings(total_node_texts, n_components=9, batch_size=32)
    # print(f"Node embeddings shape: {node_embeddings.shape}")
    
    # bond_embeddings = get_chemberta_embeddings(total_bond_texts, n_components=10, batch_size=32)
    # print(f"Bond embeddings shape: {bond_embeddings.shape}")
    