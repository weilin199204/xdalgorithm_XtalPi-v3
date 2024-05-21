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

import pyshapeit

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
    def __init__(self,conformers_num,
            failed_allowed,
            reward_weight,
            d_upper,
            d_lower,
            keep,
            pList_max_allowed,
            pharmacophore_idxs,
            template_mol,
            use_chemaxon,
            protonation,
            pharmacophore_weights):
        #super(PharmacophoreShapeCombination,self).__init__(parameters)
        self.conformers_num=conformers_num
        self.failed_allowed=failed_allowed
        self.reward_weight=reward_weight
        self.d_upper=d_upper
        self.d_lower =d_lower 
        self.keep =keep
        self.pList_max_allowed =pList_max_allowed
        self.pharmacophore_idxs = pharmacophore_idxs
        self.template_mol =template_mol
        self.use_chemaxon = use_chemaxon
        self.protonation=protonation
        self.pharmacophore_weights =pharmacophore_weights
        #self.conformers_num = self.parameters.specific_parameters["conformers_num"]
        #self.failed_allowed = self.parameters.specific_parameters["max_to_try"]
        #self.reward_weight = self.parameters.specific_parameters["reward_weights"]
        #self.d_upper = self.parameters.specific_parameters["d_upper"]
        #self.d_lower = self.parameters.specific_parameters["d_lower"]
        #self.keep = self.parameters.specific_parameters["valid_p4cores"]
        #self.pList_max_allowed = self.parameters.specific_parameters["max_num_p4cores"]
        #self.pharmacophore_idxs = self.parameters.specific_parameters["pharmacophore_idxs"]
        #self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        #self.use_chemaxon = self.parameters.specific_parameters.get('use_chemaxon', False)
        #self.protonation = self.parameters.specific_parameters.get('protonation', True)
        #if "pharmacophore_weights" in self.parameters.specific_parameters:
        #    self.pharmacophore_weights = self.parameters.specific_parameters["pharmacophore_weights"]
        #else:
        #    self.pharmacophore_weights = [1.] * len(self.pharmacophore_idxs)

        self.fdef = AllChem.BuildFeatureFactory('/home/wei/weilin/soft/xdalgorithm_XtalPi-main/xdalgorithm/toolbox/scoring_component_library/defined_BaseFeatures.fdef')
        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = fparams

        reference_rawfeats = self.fdef.GetFeaturesForMol(template_mol)
        reference_feats = [f for f in reference_rawfeats if f.GetFamily() in self.keep]
        self.reference_fms = FeatMaps.FeatMap(feats = reference_feats,weights=[1]*len(reference_feats),params=self.fmParams)
        self.prob_feats = self.fdef.GetFeaturesForMol(self.template_mol)
        self.prob_points= [list(x.GetPos()) for x in self.prob_feats]
        self.template_contrib = Chem.rdMolDescriptors._CalcCrippenContribs(self.template_mol)
        self.p4core = self._define_pharmacophore(self.pharmacophore_idxs)



    def _define_pharmacophore(self, idxs):
        required_feats = [self.prob_feats[idx] for idx in idxs if idx < len(self.prob_feats)]
        assert len(required_feats) >= 2
        pharm_core = rkcPharmacophore.Pharmacophore(required_feats)
        for idx_i, idx_j in combinations(range(len(required_feats)), 2):
            dist = rdGeometry.Point3D.Distance(required_feats[idx_i].GetPos(), required_feats[idx_j].GetPos())
            pharm_core.setLowerBound(idx_i, idx_j, min(dist - self.d_lower, 0))
            pharm_core.setUpperBound(idx_i, idx_j, dist + self.d_upper)
        return pharm_core

#    def calculate_score(self,molecules:List) -> ComponentSummary:
#        scores = self._score_molecules(molecules)
#        score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
#        return score_summary

#    def _score_molecules(self,molecules):
#        return np.array([self._score_molecule(molecule) for molecule in molecules])

    def calculate_score(self,molecule):
        score = self._score_molecules(molecule)
        return score

#    def calculate_score(self,input_mol):
    def _score_molecule(self,input_mol):
        # round0: valid check
        #input_mol = Chem.MolFromSmiles(smi)
        #if not input_mol:
        #    return 0.0
        # The valid has been checked before input. scoring functio of invalid smiles is 0.
        # round1: pharmacophore matched
        if not input_mol:
             print('XXX')
             return 0.0
        if self.protonation :
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
                shape_score=pyshapeit.AlignMol(self.template_mol,mol,"scoreOnly")
                score=pc_score * self.reward_weight[2]+ shape_score*self.reward_weight[3]

            except RuntimeError:
                print('RuntimeError')
                continue

            if score > max_feat_score:
                max_feat_score = score

        for idx,mol in enumerate(multi_temps1):
            shape_score=pyshapeit.AlignMol(self.template_mol,mol)
            rawFeats = self.fdef.GetFeaturesForMol(mol)
            featList = [f for f in rawFeats if f.GetFamily() in self.keep]
            try:
                pc_score = self.reference_fms.ScoreFeats(featList) / self.reference_fms.GetNumFeatures()
                score=pc_score * self.reward_weight[2]+ shape_score*self.reward_weight[3]

            except RuntimeError:
                print('RuntimeError')
                continue

            if score > max_feat_score:
                max_feat_score = score

        return self.reward_weight[0] + self.reward_weight[1] + max_feat_score

    def get_component_type(self):
        return "pharmacophore_shape_combination"
