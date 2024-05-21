                 work_dir='.',
                 num_docking_runs=10,
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

        protein_pdbqt_file_name = 'protein.pdbqt'
        raise NotImplementedError('please wait haha')

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



