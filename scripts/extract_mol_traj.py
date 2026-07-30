from rdkit.Chem import AllChem as Chem

import pickle
import numpy as np
import argparse

def cast_input(input_):
    if isinstance(input_, np.ndarray):
        input_ = input_.tolist()
    return input_

def get_rdkit_mol(
    fname,
    positions,
    atoms,
    bond_index,
    bond_types,
):
    mol = Chem.RWMol()
    atom_map = {}

    positions = cast_input(positions)
    atoms = cast_input(atoms)
    bond_index = cast_input(bond_index)
    bond_types = cast_input(bond_types)

    for i, atom_type in enumerate(atoms):
        atom = Chem.Atom(atom_type)
        atom_idx = mol.AddAtom(atom)
        atom_map[i] = atom_idx

    added_bonds = set()
    for bond_idx1, bond_idx2, bond_type in zip(bond_index[0], bond_index[1], bond_types):
        if (bond_idx2, bond_idx1) in added_bonds or (bond_idx1, bond_idx2) in added_bonds:
            continue
        print("Idx, type:", bond_idx1, bond_idx2, bond_type)
        added_bonds.add((bond_idx1, bond_idx2))
        print(added_bonds)
        atom1, atom2 = int(bond_idx1), int(bond_idx2)
        mol.AddBond(atom_map[atom1], atom_map[atom2], Chem.BondType.values[bond_type])
    print("Len of added_bonds:", len(added_bonds))

    conf = Chem.Conformer(mol.GetNumAtoms())
    for conf_id in positions:
        for i, pos in enumerate(conf_id):
            conf.SetAtomPosition(i, pos)
        mol.AddConformer(conf, assignId=True)
        conf = Chem.Conformer(mol.GetNumAtoms())
    Chem.MolToPDBFile(mol, f"{fname}_confs.pdb")
    print(f"Done! Saved as {fname}_confs.pdb")
    return mol


if __name__ == "__main__":

    p = argparse.ArgumentParser("Extract a trajectory from Tito sample.")
    p.add_argument("-f", required=True, type=str, help="Path to .pkl sample.py output.")
    args = p.parse_args()

    with open(args.f, "rb") as f:
        data_dict = pickle.load(f)

    filename = args.f.split("/")[-1].split(".")[0]

    # extrat out the mol
    # This saves to the current dir with output `molecule.pdb`
    print("Loading data into a Mol Object with multiple confs...")
    get_rdkit_mol(
            fname=filename,
            positions=data_dict["traj"].squeeze(0) * 10,
            atoms=data_dict["atoms"],
            bond_index=data_dict["bond_index"],
            bond_types=data_dict["bond_type"],
            )
