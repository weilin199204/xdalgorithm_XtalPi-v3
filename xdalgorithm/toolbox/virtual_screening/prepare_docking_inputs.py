import os
import numpy as np
import pandas as pd

from rdkit import Chem
import MDAnalysis as mda

class PrepareDockingInputs(object):
    def __init__(self, ligand_sdf_file_name, ligand_molecule_name, working_dir_name):
        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.ligand_molecule_name = ligand_molecule_name

        self.path_prefix = os.path.join(working_dir_name, self.ligand_molecule_name)
        os.mkdir(self.path_prefix)

        os.environ['MGLPY'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/bin/python'
        os.environ['MGLUTIL'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
        self.MGLPY = os.environ['MGLPY']
        self.MGLUTIL = os.environ['MGLUTIL']

    def __rename_ligand_pdb_atom_names__(self, input_pdb_file_name, output_pdb_file_name, ligand_resname):
        segment_group = mda.Universe(input_pdb_file_name).segments
        for current_segment in segment_group:
            current_segment.segid = ' '
            residue_group = current_segment.residues
            for current_residue in residue_group:
                current_residue.resname = ligand_resname
                current_residue.resid = 1
                atoms_in_current_res = current_residue.atoms
                for atom_idx, atom in enumerate(atoms_in_current_res):
                    atom_element = atom.element
                    atom_name = atom_element + str(atom_idx + 1)
                    atom.name = atom_name

        segment_group.atoms.write(output_pdb_file_name, remarks=None, bonds=None)

    def __get_pdb_from_sdf__(self, sdf_file_name, path_prefix, output_prefix, ligand_resname):
        mol_list = []
        output_pdb_file_name_list = []
        sdf_iterator = Chem.SDMolSupplier(sdf_file_name, removeHs=False)
        for idx, mol in enumerate(sdf_iterator):
            current_temp_pdb_file_name = os.path.join(path_prefix, output_prefix + '_temp_conf_' + str(idx) + '.pdb')
            current_output_pdb_file_name = os.path.join(path_prefix, output_prefix + '_conf_' + str(idx) + '.pdb')
            Chem.MolToPDBFile(mol, current_temp_pdb_file_name)
            self.__rename_ligand_pdb_atom_names__(current_temp_pdb_file_name, current_output_pdb_file_name, ligand_resname)
            mol_list.append(mol)
            output_pdb_file_name_list.append(current_output_pdb_file_name)

        return mol_list, output_pdb_file_name_list

    def prepare_ligand_pdbqt_file(self):
        mol_list, output_pdb_file_name_list = self.__get_pdb_from_sdf__(self.ligand_sdf_file_name, self.path_prefix, self.ligand_molecule_name, 'MOL')
        num_conformations = len(output_pdb_file_name_list)
        conf_pdbqt_file_name_list = [None] * num_conformations

        os.environ['PYTHONPATH'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs'

        for conf_idx in range(num_conformations):
            conf_pdb_file_prefix = output_pdb_file_name_list[conf_idx].split('.')[0]
            conf_pdb_file_name = conf_pdb_file_prefix + '.pdb'
            conf_pdbqt_file_name = conf_pdb_file_prefix + '.pdbqt'

            os.system("%s %s/prepare_ligand4.py -l %s -o %s" %(self.MGLPY, self.MGLUTIL, conf_pdb_file_name, conf_pdbqt_file_name))
            conf_pdbqt_file_name_list[conf_idx] = os.path.abspath(conf_pdbqt_file_name)

        return conf_pdbqt_file_name_list
