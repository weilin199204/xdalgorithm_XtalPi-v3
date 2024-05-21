from typing import List
import os
import random
import numpy as np
import copy
from collections import defaultdict
from itertools import combinations

from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary


from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import RDConfig
from rdkit.Chem import rdGeometry
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D import Pharmacophore as rkcPharmacophore
from rdkit.Chem.Pharm3D.EmbedLib import EmbedPharmacophore
from rdkit.Chem.Pharm3D.EmbedLib import MatchPharmacophoreToMol
from rdkit.Chem.Pharm3D.EmbedLib import GetAllPharmacophoreMatches
from rdkit.Numerics import rdAlignment
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw


import random
import time

from rdkit.Chem import rdMolAlign
# template_mol_file:'2JKM-ligand-model-bi9.sdf'
# d_upper: 1.5
# d_lower: 0.5
# keep:('Donor','Acceptor','NegIonizable','PosIonizable','Aromatic')
# conformers_num:2
# pList_max_allowed: 30
# failed_allowed
# pharmacophore_idx: (0,6,7,8,9)
#reward_weight:(0.1,0.25,0.5,0.5)


class PharmacophoreShapeCombination(BaseScoreComponent):
    def __init__(self,parameters: ComponentParameters):
        super(Pharmacophore_Align, self).__init__(parameters)
        self.d_upper = self.parameters.specific_parameters["d_upper"]

        os.environ['MGLPY'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/bin/python'
        os.environ['MGLUTIL'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
        self.MGLPY = os.environ['MGLPY']
        self.MGLUTIL = os.environ['MGLUTIL']
        sdlf.docking_conf=self.parameters.specific_parameters["docking_conf"]

    def prepare_ligand_pdbqt_file(self):
        mol_list, output_pdb_file_name_list = self.__get_pdb_from_sdf__(self.ligand_sdf_file_name, self.path_prefix, self.ligand_molecule_name, 'MOL')
        num_conformations = len(output_pdb_file_name_list)
        conf_pdbqt_file_name_list = [None] * num_conformations

        os.environ['PYTHONPATH'] = '/data/aidd-server/Modules/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs'

      
         conf_pdb_file_name = conf_pdb_file_prefix + '.pdb'
         conf_pdbqt_file_name = conf_pdb_file_prefix + '.pdbqt'

         os.system("%s %s/prepare_ligand4.py -l %s -o %s" %(self.MGLPY, self.MGLUTIL, conf_pdb_file_name, conf_pdbqt_file_name))
         conf_pdbqt_file_name_list[conf_idx] = os.path.abspath(conf_pdbqt_file_name)

         return conf_pdbqt_file_name_list
 
      def __write_ligand_batch_file__(self, protein_map_fld_file_name, output_prefix_list, conf_pdbqt_file_name_list, ligand_batch_file_name):
        num_ligand_confs = len(output_prefix_list)
        with open(ligand_batch_file_name, 'w') as ligand_batch_file:
            for idx in range(num_ligand_confs):
                ligand_batch_file.write(protein_map_fld_file_name + '\n')
                ligand_batch_file.write(conf_pdbqt_file_name_list[idx] + '\n')
                ligand_batch_file.write(output_prefix_list[idx] + '\n')




    def calculate_score(self,molecules:List) -> ComponentSummary:
        scores = self._score_molecules(molecules)
        score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
        return score_summary

    def _score_molecules(self,molecules):
        return np.array([self._score_molecule(molecule) for molecule in molecules])



    def _score_molecule(self,input_mol):
        # round0: valid check
        #input_mol = Chem.MolFromSmiles(smi)
        #if not input_mol:
        #    return 0.0
        # The valid has been checked before input. scoring functio of invalid smiles is 0.
        # round1: pharmacophore matched
    def __prepare_protein_grid_files__(self):





	if not input_mol :
		return 0.0

	smi=Chem.MolToSmiles(input_mol)
	name_rad=random.random()
	name_time=time.time()
	subprocess.call('ligprep -epik -ph 7.4 -s 1  -ismi '+smi+'-osd /backup/wei/tmp/'+str(name_rad)+str(name_time)+'.sdf')
	input_mol=Chem.MolFromMolFile('/backup/wei/tmp/'+str(name_rad)+str(name_time)+'.sdf')

		
	match, mList = MatchPharmacophoreToMol(input_mol, self.fdef, self.p4core)
	if not match:
		return self.reward_weight[0]

        # round2: pharmacophore distances
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(input_mol)
        pList = GetAllPharmacophoreMatches(mList, bounds, self.p4core)
        if len(pList) == 0:  # if failed
            return self.reward_weight[1]+self.reward_weight[0]


        if len(pList) > self.pList_max_allowed:
            random_idxs = list(range(len(pList)))
            random.shuffle(random_idxs)
            pList = [pList[i] for i in random_idxs[:self.pList_max_allowed]]
        phMatches = []
        for p_idx,p in enumerate(pList):
            num_feature = len(p)
            phMatch = []
            for j in range(num_feature):
                phMatch.append(p[j].GetAtomIds())
            phMatches.append(phMatch)
        res = []
        for phMatch in phMatches:
            bm,embeds,nFail = EmbedPharmacophore(input_mol,phMatch,self.p4core,count=self.failed_allowed,silent=1)
            if nFail < self.failed_allowed:
                for embed in embeds:
                    AllChem.UFFOptimizeMolecule(embed)
                    m = copy.deepcopy(embed)
                    if m is None:
                        continue
                    if len(res)==0:
                        res.append(m)
                    else:
                        add_m = True
                        for r in res:
                            if AllChem.GetBestRMS(r,m) < 0.1:
                                add_m = False;
                                break
                        if add_m:
                            res.append(m)
                            break
        temp_conf_collect = res
        p = AllChem.ETKDGv2()
        p.verbose = False
        multi_temps1 = []
        for temp in temp_conf_collect:
            multi_temps1.append(Chem.AddHs(copy.deepcopy(temp)))
        for mol in multi_temps1:
            AllChem.EmbedMultipleConfs(mol,self.conformers_num, p)
        crippen_contribs = [Chem.rdMolDescriptors._CalcCrippenContribs(mol) for mol in multi_temps1]

        for idx,mol in enumerate(multi_temps1):
            for cid in range(self.conformers_num):
                try:
                    crippenO3A = rdMolAlign.GetCrippenO3A(mol, self.template_mol, crippen_contribs[idx], self.template_contrib, cid, 0)
                    crippenO3A.Align()
                except ValueError:
                    print('ValueError')
                    continue
        max_feat_score = 0
        for idx,mol in enumerate(multi_temps1):
            rawFeats = self.fdef.GetFeaturesForMol(mol)
            featList = [f for f in rawFeats if f.GetFamily() in self.keep]
            try:
                pc_score = self.reference_fms.ScoreFeats(featList) / self.reference_fms.GetNumFeatures()
		shape_score=cpyshapeit.AlignMol(self.template_mol,mol)
		score=pc_score * self.reward_weight[2]+ shape_score*self.reward_weight[3]

            except RuntimeError:
                print('RuntimeError')
                continue

            if score > max_feat_score:
                max_feat_score = score

        return self.reward_weight[0] + self.reward_weight[1] + max_feat_score

    def get_component_type(self):
        return "pharmacophore_align"
