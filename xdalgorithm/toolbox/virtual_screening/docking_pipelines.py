import os
import traceback
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D
import MDAnalysis as mda

class DockingPipelines(object):
    def __init__(self,
                 database_info_df,
                 receptor_pdbqt_file_name,
                 work_dir='.',
                 bound_ligand_pdbqt_file_name=None,
                 num_docking_poses=10,
                 target_center=None,
                 num_grid_points=(40, 40, 40),
                 grid_spacing=(0.375, 0.375, 0.375)):

        self.database_info_df = deepcopy(database_info_df)
        self.work_dir = work_dir

        if self.work_dir != '.':
            os.system("cp %s %s" %(receptor_pdbqt_file_name, self.work_dir))

        os.chdir(self.work_dir)

        if '/' in receptor_pdbqt_file_name:
            os.system("cp %s ." %(receptor_pdbqt_file_name))

        self.receptor_pdbqt_file_name = os.path.basename(receptor_pdbqt_file_name)
        self.bound_ligand_pdbqt_file_name = bound_ligand_pdbqt_file_name
        self.num_docking_poses = num_docking_poses
        self.target_center = target_center
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing

        if self.bound_ligand_pdbqt_file_name is None and self.target_center is None:
            raise ValueError('bound_ligand_pdbqt_file_names and target_center should have at least one specified.')

        if self.target_center:
            self.target_center_x = self.target_center[0]
            self.target_center_y = self.target_center[1]
            self.target_center_z = self.target_center[2]

        os.environ['MGLPY'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/bin/python'
        os.environ['MGLUTIL'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
        self.MGLPY = os.environ['MGLPY']
        self.MGLUTIL = os.environ['MGLUTIL']

        self.lock = mp.Lock()

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

    def __get_pdb_from_sdf__(self, sdf_file_name, output_prefix, ligand_resname):
        mol_list = []
        output_pdb_file_name_list = []
        sdf_iterator = Chem.SDMolSupplier(sdf_file_name, removeHs=False)
        for idx, mol in enumerate(sdf_iterator):
            current_temp_pdb_file_name = output_prefix + '_temp_conf_' + str(idx) + '.pdb'
            current_output_pdb_file_name = output_prefix + '_conf_' + str(idx) + '.pdb'
            Chem.MolToPDBFile(mol, current_temp_pdb_file_name)
            self.__rename_ligand_pdb_atom_names__(current_temp_pdb_file_name, current_output_pdb_file_name, ligand_resname)
            mol_list.append(mol)
            output_pdb_file_name_list.append(current_output_pdb_file_name)

        return mol_list, output_pdb_file_name_list

    def __get_docking_pose_conformation_sdf__(self, reference_mol, reference_pdb_file_name, docked_pdb_file_name, docked_sdf_file_name):
        reference_ag = mda.Universe(reference_pdb_file_name).atoms
        reference_atom_names = reference_ag.names
        docked_mol = deepcopy(reference_mol)
        docked_mol_conformer = docked_mol.GetConformer()
        docked_conf_ag = mda.Universe(docked_pdb_file_name).atoms
        docked_conf_atom_names = docked_conf_ag.names
        docked_conf_atom_coords = docked_conf_ag.positions
        num_docked_conf_atoms = docked_conf_ag.n_atoms

        for idx in range(num_docked_conf_atoms):
            current_docked_conf_atom_coord = docked_conf_atom_coords[idx, :].astype(np.float64)
            current_docked_conf_atom_coord_point_3D = Point3D(current_docked_conf_atom_coord[0], current_docked_conf_atom_coord[1], current_docked_conf_atom_coord[2])
            current_docked_conf_atom_name = docked_conf_atom_names[idx]
            current_docked_conf_atom_idx = int(np.where(reference_atom_names == current_docked_conf_atom_name)[0][0])
            docked_mol_conformer.SetAtomPosition(current_docked_conf_atom_idx, current_docked_conf_atom_coord_point_3D)

        docked_mol_no_h = Chem.RemoveHs(docked_mol)
        docked_mol_added_h = Chem.AddHs(docked_mol_no_h, addCoords=True)

        sdf_writer = Chem.SDWriter(docked_sdf_file_name)
        sdf_writer.write(docked_mol_added_h)
        sdf_writer.flush()
        sdf_writer.close()

    def __parse_virtual_screening_pdbqt_file__(self, pdbqt_file_name):
        with open(pdbqt_file_name, 'r') as f:
            pdbqt_lines = f.read().strip().split('\n')

        for line_idx, line in enumerate(pdbqt_lines):
            if 'lig_hb_atoms' in line:
                num_hydrogen_bonds = int(line.split()[4])
            elif 'pi_pi' in line:
                num_pi_pi_interactions = int(line.split()[3])
            elif 'lig_close_ats' in line:
                num_close_atoms = int(line.split()[3])
            elif 'rmsd, LE' in line:
                cluster_info_line_idx = line_idx + 1

        ligand_efficiency = float(pdbqt_lines[cluster_info_line_idx].split()[2].split(',')[1])

        if not 'num_pi_pi_interactions' in locals():
            num_pi_pi_interactions = 0

        return num_hydrogen_bonds, num_pi_pi_interactions, num_close_atoms, ligand_efficiency

    def __prepare_receptor_pdbqt_file__(self):
        raise NotImplementedError

    def __docking_process__(self,
    						mol_info_dict,
                            docking_pose_sdf_path_list,
                            docking_pose_ligand_efficiency_list,
                            docking_pose_num_close_atoms_list,
                            docking_pose_num_hydrogen_bonds_list,
                            docking_pose_num_pi_pi_interactions_list):

        dir_name_tuple = mol_info_dict['dir_name_tuple']
        conf_pdb_file_prefix = mol_info_dict['conf_pdb_file_prefix']
        conf_mol = mol_info_dict['conf_mol']
        num_docking_run = mol_info_dict['num_docking_run']
        process_idx = mol_info_dict['process_idx']

        ligand_dir_name = dir_name_tuple[0]
        conf_dir_name = dir_name_tuple[1]
        os.chdir(ligand_dir_name + '/' + conf_dir_name)

        conf_pdb_file_name = '../' + conf_pdb_file_prefix + '.pdb'
        conf_pdbqt_file_name = conf_pdb_file_prefix + '.pdbqt'
        possible_docking_pose_pdbqt_file_name_list = [conf_pdb_file_prefix + '_vs.pdbqt', conf_pdb_file_prefix + '_vs_le.pdbqt']

        os.system("ln -s ../../%s ." %(self.receptor_pdbqt_file_name))
        os.system("ln -s ../../receptor*map* .")
        os.system("%s %s/prepare_ligand4.py -l %s -o %s" %(self.MGLPY, self.MGLUTIL, conf_pdb_file_name, conf_pdbqt_file_name))

        with self.lock:
        	os.system("/data/aidd-server/Modules/AutoDock-GPU/bin/autodock_gpu_256wi \
                      -ffile receptor.maps.fld \
                      -lfile %s -nrun %d" %(conf_pdbqt_file_name, num_docking_run))

        try:
            os.system("%s %s/process_VSResults.py -d . -r %s -B -D -p" %(self.MGLPY, self.MGLUTIL, self.receptor_pdbqt_file_name))

            for possible_docking_pose_pdbqt_file_name in possible_docking_pose_pdbqt_file_name_list:
                if os.path.exists(possible_docking_pose_pdbqt_file_name):
                    docking_pose_pdbqt_file_name = possible_docking_pose_pdbqt_file_name
                    break

            os.system("%s %s/pdbqt_to_pdb.py -f %s -o docking_pose_init.pdb" %(self.MGLPY, self.MGLUTIL, docking_pose_pdbqt_file_name))

            self.__get_docking_pose_conformation_sdf__(conf_mol, conf_pdb_file_name, 'docking_pose_init.pdb', 'docking_pose.sdf')
            num_hydrogen_bonds, num_pi_pi_interactions, num_close_atoms, ligand_efficiency = self.__parse_virtual_screening_pdbqt_file__(docking_pose_pdbqt_file_name)

            docking_pose_sdf_path_list[process_idx] = os.path.abspath('docking_pose.sdf')
            docking_pose_ligand_efficiency_list[process_idx] = ligand_efficiency
            docking_pose_num_close_atoms_list[process_idx] = num_close_atoms
            docking_pose_num_hydrogen_bonds_list[process_idx] = num_hydrogen_bonds
            docking_pose_num_pi_pi_interactions_list[process_idx] = num_pi_pi_interactions

        except Exception:
            print('Error occurred in ' + ligand_dir_name + '/' + conf_dir_name)
            print(traceback.format_exc())

    def run_docking_protocol(self):
        ligand_unique_id_array = self.database_info_df.loc[:, 'unique_id'].values
        ligand_sdf_file_name_array = self.database_info_df.loc[:, 'sdf_path'].values.astype('U')
        num_ligands = ligand_unique_id_array.shape[0]

        docking_pose_unique_id_list = []

        os.environ['PYTHONPATH'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs'
        if 'LD_LIBRARY_PATH' in os.environ:
            os.environ['LD_LIBRARY_PATH'] += ':/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/lib'
        else:
            os.environ['LD_LIBRARY_PATH'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/lib'

        os.system("%s %s/prepare_gpf4.py -r %s \
                  -p ligand_types='H, HD, C, A, N, NA, OA, F, P, SA, S, Cl, Br, I' \
                  -p npts='%d, %d, %d' -p spacing='%f, %f, %f' \
                  -p gridcenter='%f, %f, %f' \
                  -o receptor.gpf" %(self.MGLPY,  self.MGLUTIL, self.receptor_pdbqt_file_name, self.num_grid_points[0], self.num_grid_points[1], self.num_grid_points[2], self.grid_spacing[0], self.grid_spacing[1], self.grid_spacing[2], self.target_center_x, self.target_center_y, self.target_center_z))

        os.system("/data/aidd-server/Modules/AutoDock/autogrid4 -p receptor.gpf -l receptor.glg")

        ligand_conf_info_dict = {}
        dir_name_tuple_list = []
        current_process_idx = -1
        for ligand_idx in range(num_ligands):
            ligand_unique_id = ligand_unique_id_array[ligand_idx]
            ligand_sdf_file_name = ligand_sdf_file_name_array[ligand_idx]

            ligand_dir_name = 'ligand_' + str(ligand_idx)
            if not os.path.isdir(ligand_dir_name):
                os.mkdir(ligand_dir_name)
            os.chdir(ligand_dir_name)

            mol_list, output_pdb_file_name_list = self.__get_pdb_from_sdf__(ligand_sdf_file_name, ligand_dir_name, 'MOL')
            num_conformations_list = []
            num_conformations = len(output_pdb_file_name_list)
            num_docking_run = int(np.ceil(self.num_docking_poses / num_conformations))
            num_conformations_list.append(num_conformations)

            for conf_idx in range(num_conformations):
                conf_mol = mol_list[conf_idx]
                conf_dir_name = 'conf_' + str(conf_idx)
                conf_pdb_file_prefix = output_pdb_file_name_list[conf_idx].split('.')[0]

                current_process_idx += 1
                dir_name_tuple = (ligand_dir_name, conf_dir_name)
                dir_name_tuple_list.append(dir_name_tuple)
                ligand_conf_info_dict[dir_name_tuple] = {}
                ligand_conf_info_dict[dir_name_tuple]['dir_name_tuple'] = dir_name_tuple
                ligand_conf_info_dict[dir_name_tuple]['conf_mol'] = conf_mol
                ligand_conf_info_dict[dir_name_tuple]['num_docking_run'] = num_docking_run
                ligand_conf_info_dict[dir_name_tuple]['conf_pdb_file_prefix'] = conf_pdb_file_prefix
                ligand_conf_info_dict[dir_name_tuple]['process_idx'] = current_process_idx
                docking_pose_unique_id_list.append(ligand_unique_id)
                os.mkdir(conf_dir_name)

            os.chdir('..')

        print(ligand_conf_info_dict)

        num_process = current_process_idx + 1

        manager = mp.Manager()
        docking_pose_sdf_path_list = manager.list()
        docking_pose_ligand_efficiency_list = manager.list()
        docking_pose_num_close_atoms_list = manager.list()
        docking_pose_num_hydrogen_bonds_list = manager.list()
        docking_pose_num_pi_pi_interactions_list = manager.list()

        docking_pose_sdf_path_list.extend([None] * num_process)
        docking_pose_ligand_efficiency_list.extend([None] * num_process)
        docking_pose_num_close_atoms_list.extend([None] * num_process)
        docking_pose_num_hydrogen_bonds_list.extend([None] * num_process)
        docking_pose_num_pi_pi_interactions_list.extend([None] * num_process)

        docking_process_list = []

        for dir_name_tuple in dir_name_tuple_list:
            docking_process = mp.Process(target=self.__docking_process__,
                                         args=(ligand_conf_info_dict[dir_name_tuple],
                                               docking_pose_sdf_path_list,
                                               docking_pose_ligand_efficiency_list,
                                               docking_pose_num_close_atoms_list,
                                               docking_pose_num_hydrogen_bonds_list,
                                               docking_pose_num_pi_pi_interactions_list))

            docking_process_list.append(docking_process)

        for docking_process in docking_process_list:
            docking_process.start()

        for docking_process in docking_process_list:
            docking_process.join()

        docking_pose_unique_id_array = np.array(docking_pose_unique_id_list).astype(np.int64)
        docking_pose_sdf_path_array = np.array(docking_pose_sdf_path_list).astype('U')
        docking_pose_ligand_efficiency_array = np.array(docking_pose_ligand_efficiency_list).astype(np.float64)
        docking_pose_num_close_atoms_array = np.array(docking_pose_num_close_atoms_list).astype(np.int32)
        docking_pose_num_hydrogen_bonds_array = np.array(docking_pose_num_hydrogen_bonds_list).astype(np.int32)
        docking_pose_num_pi_pi_interactions_array = np.array(docking_pose_num_pi_pi_interactions_list).astype(np.int32)

        docking_pose_info_dict = {'unique_id': docking_pose_unique_id_array,
                                  'sdf_path': docking_pose_sdf_path_array,
                                  'ligand_efficiency': docking_pose_ligand_efficiency_array,
                                  'num_close_atoms': docking_pose_num_close_atoms_array,
                                  'num_hydrogen_bonds': docking_pose_num_hydrogen_bonds_array,
                                  'num_pi_pi_interactions': docking_pose_num_pi_pi_interactions_array}

        docking_pose_info_df = pd.DataFrame(docking_pose_info_dict)
        return docking_pose_info_df
