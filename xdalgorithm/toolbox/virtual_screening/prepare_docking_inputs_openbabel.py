import os

import MDAnalysis as mda
from rdkit import Chem
from openbabel import pybel

class PrepareDockingInputs(object):
    def __init__(self, ligand_sdf_file_name, ligand_molecule_name, working_dir_name):
        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.ligand_molecule_name = ligand_molecule_name

        self.path_prefix = os.path.join(working_dir_name, self.ligand_molecule_name)
        os.mkdir(self.path_prefix)

    def __rename_ligand_pdb_atom_names__(self, input_pdb_file_name, output_pdb_file_name, ligand_resname):
        segment_group = mda.Universe(input_pdb_file_name).segments
        for current_segment in segment_group:
            current_segment.segid = 'A'
            residue_group = current_segment.residues
            for current_residue in residue_group:
                current_residue.resname = ligand_resname
                current_residue.resid = 1
                atoms_in_current_res = current_residue.atoms
                for atom_idx, atom in enumerate(atoms_in_current_res):
                    atom_element = atom.element
                    atom_name = atom_element + str(atom_idx + 1)
                    atom.name = atom_name

        segment_group.atoms.write(output_pdb_file_name)

    def __prepared_pdb_bond_orders__(self, input_rdkit_pdb_file_name, input_mda_pdb_file_name, output_prepared_pdb_file_name):
        rdkit_conect_records_list = []
        mda_cryst_records_list = []
        mda_atom_records_list = []
        mda_end_records_list = []

        with open(input_rdkit_pdb_file_name, 'r') as rdkit_pdb_file:
            rdkit_pdb_lines = rdkit_pdb_file.readlines()

        with open(input_mda_pdb_file_name, 'r') as mda_pdb_file:
            mda_pdb_lines = mda_pdb_file.readlines()

        for rdkit_pdb_line in rdkit_pdb_lines:
            if rdkit_pdb_line.startswith('CONECT'):
                rdkit_conect_records_list.append(rdkit_pdb_line)

        for mda_pdb_line in mda_pdb_lines:
            if mda_pdb_line.startswith('CRYST1'):
                mda_cryst_records_list.append(mda_pdb_line)
            elif mda_pdb_line.startswith('ATOM'):
                mda_atom_records_list.append(mda_pdb_line)
            elif mda_pdb_line.startswith('END'):
                mda_end_records_list.append(mda_pdb_line)

        total_records_list = mda_cryst_records_list + mda_atom_records_list + rdkit_conect_records_list + mda_end_records_list

        with open(output_prepared_pdb_file_name, 'w') as prepared_pdb_file:
            for prepared_pdb_line in total_records_list:
                prepared_pdb_file.write(prepared_pdb_line)

    def prepare_ligand_pdbqt_file(self):
        sdf_iterator = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)
        num_conformations = len(sdf_iterator)
        conf_pdbqt_file_name_list = [None] * num_conformations

        for conf_idx, conf_rdmol in enumerate(sdf_iterator):
            conf_rdkit_pdb_file_name = os.path.join(self.path_prefix, self.ligand_molecule_name + '_conf_' + str(conf_idx) + '_rdkit.pdb')
            conf_mda_pdb_file_name = os.path.join(self.path_prefix, self.ligand_molecule_name + '_conf_' + str(conf_idx) + '_mda.pdb')
            conf_prepared_pdb_file_name = os.path.join(self.path_prefix, self.ligand_molecule_name + '_conf_' + str(conf_idx) + '_prepared.pdb')
            conf_pdbqt_file_name = os.path.join(self.path_prefix, self.ligand_molecule_name + '_conf_' + str(conf_idx) + '.pdbqt')

            Chem.MolToPDBFile(conf_rdmol, conf_rdkit_pdb_file_name, flavor=4)
            self.__rename_ligand_pdb_atom_names__(conf_rdkit_pdb_file_name, conf_mda_pdb_file_name, 'MOL')
            self.__prepared_pdb_bond_orders__(conf_rdkit_pdb_file_name, conf_mda_pdb_file_name, conf_prepared_pdb_file_name)

            conf_obmol = next(pybel.readfile('pdb', conf_prepared_pdb_file_name))
            _ = conf_obmol.calccharges('gasteiger')
            conf_obmol.write('pdbqt', conf_pdbqt_file_name, overwrite=True)

            conf_pdbqt_file_name_list[conf_idx] = os.path.abspath(conf_pdbqt_file_name)

        return conf_pdbqt_file_name_list
