from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_geometric.data import Data
from collections import deque
from torch_geometric.utils import to_networkx
import re


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
        
def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)
    
def get_smiles(mol):
    smiles = mol2smiles(mol)
    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            return smiles
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
            return None
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
            return None
    else:
        return None

def seq_to_mol(seq_str):
    tokens = seq_str.split()
    mol = Chem.RWMol()

    ctx_start = tokens.index('<boc>') + 1
    ctx_end = tokens.index('<eoc>')
    ctx_tokens = tokens[ctx_start:ctx_end+1]

    id_atom_lookup = {}
    for i in range(0, len(ctx_tokens), 3):
        atom_type = ctx_tokens[i]
        atom_id = ctx_tokens[i + 1]
        atomic_symbol = atom_type.split('_')[1]
        atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
        mol.AddAtom(Chem.Atom(atomic_num))
        id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1

    # Extract bond tokens
    bond_start = tokens.index('<bog>') + 1
    bond_end = tokens.index('<eog>')
    bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']

    for i in range(0, len(bond_tokens), 3):
        src_id = bond_tokens[i]
        dest_id = bond_tokens[i + 1]
        bond_type = bond_tokens[i + 2]
        bond_type_rdkit = {
            'BOND_SINGLE': rdchem.BondType.SINGLE,
            'BOND_DOUBLE': rdchem.BondType.DOUBLE,
            'BOND_TRIPLE': rdchem.BondType.TRIPLE,
            'BOND_AROMATIC': rdchem.BondType.AROMATIC
        }[bond_type]
        
        if src_id in id_atom_lookup and dest_id in id_atom_lookup:
            mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)

    return mol

def seq_to_molecule_with_partial_charges(seq_str):
    ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    tokens = seq_str.split()
    mol = Chem.RWMol()

    ctx_start = tokens.index('<boc>') + 1
    ctx_end = tokens.index('<eoc>')
    ctx_tokens = tokens[ctx_start:ctx_end+1]

    id_atom_lookup = {}
    for i in range(0, len(ctx_tokens), 3):
        atom_type = ctx_tokens[i]
        atom_id = ctx_tokens[i + 1]
        atomic_symbol = atom_type.split('_')[1]
        atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
        mol.AddAtom(Chem.Atom(atomic_num))
        id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1

    # Extract bond tokens
    bond_start = tokens.index('<bog>') + 1
    bond_end = tokens.index('<eog>')
    bond_tokens = [token for token in tokens[bond_start:bond_end] if token != '<sepg>']

    for i in range(0, len(bond_tokens), 3):
        src_id = bond_tokens[i]
        dest_id = bond_tokens[i + 1]
        bond_type = bond_tokens[i + 2]
        bond_type_rdkit = {
            'BOND_SINGLE': rdchem.BondType.SINGLE,
            'BOND_DOUBLE': rdchem.BondType.DOUBLE,
            'BOND_TRIPLE': rdchem.BondType.TRIPLE,
            'BOND_AROMATIC': rdchem.BondType.AROMATIC
        }[bond_type]
        
        if src_id in id_atom_lookup and dest_id in id_atom_lookup:
            mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

def randperm_node(x, edge_index):
    num_nodes = x.shape[0]

    perm = torch.randperm(num_nodes)

    # Create a mapping from old node indices to new node indices
    mapping = torch.empty_like(perm)
    mapping[perm] = torch.arange(num_nodes)

    # Permute node features
    new_x = x[perm]
    # Update edge indices using the mapping
    new_edge_index = mapping[edge_index]

    return new_x, new_edge_index

def bfs_with_all_edges(G, source):
    visited = set()
    edges = set()
    edges_bfs = []

    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()
        for neighbor in G[node]:
            if neighbor not in visited:
                edges.add(tuple(sorted((node, neighbor))))
                edges_bfs.append((node, neighbor))

                visited.add(neighbor)
                queue.append(neighbor)
            else:
                if tuple(sorted((neighbor, node))) not in edges:
                    edges.add(tuple(sorted((neighbor, node))))
                    edges_bfs.append((node, neighbor))

    return  edges_bfs

def to_seq_by_bfs(data, atom_type, bond_type):
    
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)
    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}'] for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    
    G = to_networkx(data)

    #get edge order from dfs,begin from node 0, G is nx graph
    # _,edges_order_dfs = dfs_with_all_edges(G,0)
    edges_order_bfs = bfs_with_all_edges(G,0)
    for selected_source_node_idx, selected_dest_node_idx in edges_order_bfs:
        #get_edge_attr
        edge_mask = ((data.edge_index[0] == selected_source_node_idx) & (data.edge_index[1] == selected_dest_node_idx)) | \
            ((data.edge_index[0] == selected_dest_node_idx) & (data.edge_index[1] == selected_source_node_idx))  
        edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        if len(edge_indices) > 0:
            removed_edge_type = data.edge_attr[edge_indices][0].argmax().item()
        outputs.append(['<sepg>', f'IDX_{selected_source_node_idx}', f'IDX_{selected_dest_node_idx}', bond_type[removed_edge_type-1]])

    ctx[0] = '<boc>'
    ctx.append('<eoc>')
    outputs = sum(outputs,[])
    outputs[0] = '<bog>'
    outputs.append('<eog>')
    return {"text": [" ".join(ctx + outputs)]}
