import os
import filecmp
from copy import deepcopy

class AutoDockVirtualScreening(object):
    def __init__(self,
                 database_info_df,
                 protein_pdb_file_name=None,
                 protein_pdbqt_file_name=None,
                 work_dir='.',
                 num_docking_runs=10,
                 target_center=None,
                 num_grid_points=(40, 40, 40),
                 grid_spacing=(0.375, 0.375, 0.375)):

        self.database_info_df = deepcopy(database_info_df)
        self.work_dir = work_dir
        os.chdir(self.work_dir)

        if protein_pdbqt_file_name is None:
            if protein_pdb_file_name is not None:
                self.__prepare_protein_pdbqt_file__(protein_pdb_file_name)
            else:
                raise ValueError('protein_pdb_file_name or protein_pdbqt_file_name should have at least one specified.')
        else:
            if not (os.path.exists('protein.pdbqt') and filecmp.cmp(protein_pdbqt_file_name, 'protein.pdbqt', shallow=False)):
                os.system("cp %s protein.pdbqt" %(protein_pdbqt_file_name))

        self.num_docking_runs = num_docking_runs
        self.target_center = target_center
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing

        if self.target_center is None:
            raise ValueError('target_center should be specified.')

        os.environ['MGLPY'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/bin/python'
        os.environ['MGLUTIL'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
        self.MGLPY = os.environ['MGLPY']
        self.MGLUTIL = os.environ['MGLUTIL']

    def __prepare_protein_pdbqt_file__(self, protein_pdb_file_name):
        protein_pdbqt_file_name = 'protein.pdbqt'
        raise NotImplementedError('please wait haha')

    def __prepare_protein_grid_files__(self):
        os.environ['PYTHONPATH'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs'
        os.system("%s %s/prepare_gpf4.py -r protein.pdbqt \
                  -p ligand_types='H, HD, C, A, N, NA, OA, F, P, SA, S, Cl, Br, I' \
                  -p npts='%d, %d, %d' -p spacing='%f, %f, %f' \
                  -p gridcenter='%f, %f, %f' \
                  -o protein.gpf" %(self.MGLPY,
                                     self.MGLUTIL,
                                     self.num_grid_points[0],
                                     self.num_grid_points[1],
                                     self.num_grid_points[2],
                                     self.grid_spacing[0],
                                     self.grid_spacing[1],
                                     self.grid_spacing[2],
                                     self.target_center[0],
                                     self.target_center[1],
                                     self.target_center[2]))

        os.system("/data/aidd-server/Modules/AutoDock/autogrid4 -p protein.gpf -l protein.glg")

    def __write_ligand_batch_file__(self, protein_map_fld_file_name, output_prefix_list, conf_pdbqt_file_name_list, ligand_batch_file_name):
        num_ligand_confs = len(output_prefix_list)
        with open(ligand_batch_file_name, 'w') as ligand_batch_file:
            for idx in range(num_ligand_confs):
                ligand_batch_file.write(protein_map_fld_file_name + '\n')
                ligand_batch_file.write(conf_pdbqt_file_name_list[idx] + '\n')
                ligand_batch_file.write(output_prefix_list[idx] + '\n')

    def __perform_docking__(self, ligand_batch_file_name, num_docking_runs):
        os.system("/data/aidd-server/Modules/AutoDock-GPU/bin/autodock_gpu_128wi \
                   -filelist %s \
                   -nrun %d" %(ligand_batch_file_name, num_docking_runs))

    def run_batch_docking(self):
        self.__prepare_protein_grid_files__()
        ligand_unique_id_array = self.database_info_df.loc[:, 'unique_id'].values

        for ligand_unique_id in ligand_unique_id_array:
            ligand_name = 'ligand_' + str(ligand_unique_id)
            ligand_database_info_df = deepcopy(self.database_info_df.iloc[(self.database_info_df.loc[:, 'unique_id'] == ligand_unique_id).values, :])
            source_ligand_pdbqt_file_name_list = ligand_database_info_df.loc[:, 'conf_pdbqt_path_list'].values[0]
            num_conformations = len(source_ligand_pdbqt_file_name_list)
            ligand_conf_name_list = [None] * num_conformations
            ligand_pdbqt_file_name_list = [None] * num_conformations
            ligand_dlg_file_name_list = [None] * num_conformations

            os.mkdir(ligand_name)
            os.chdir(ligand_name)

            os.system("ln -s ../protein*map* .")

            for conf_idx in range(num_conformations):
                source_ligand_pdbqt_file_name = source_ligand_pdbqt_file_name_list[conf_idx]
                os.system("ln -s %s ." %(source_ligand_pdbqt_file_name))

                ligand_pdbqt_file_name = os.path.basename(source_ligand_pdbqt_file_name)
                ligand_conf_name = ligand_pdbqt_file_name.split('.')[0]
                ligand_conf_name_list[conf_idx] = ligand_conf_name
                ligand_pdbqt_file_name_list[conf_idx] = ligand_pdbqt_file_name
                ligand_dlg_file_name_list[conf_idx] = ligand_conf_name + '.dlg'

            self.__write_ligand_batch_file__('protein.maps.fld', ligand_conf_name_list, ligand_pdbqt_file_name_list, 'ligand_conf_batch.dat')
            self.__perform_docking__('ligand_conf_batch.dat', self.num_docking_runs)

            for conf_idx in range(num_conformations):
                ligand_conf_name = ligand_conf_name_list[conf_idx]
                ligand_dlg_file_name = ligand_dlg_file_name_list[conf_idx]
                if not os.path.exists(ligand_dlg_file_name):
                    raise Exception('error in ' + ligand_conf_name)
                else:
                    continue

            os.chdir('..')
