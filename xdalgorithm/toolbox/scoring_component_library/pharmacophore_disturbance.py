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
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdChemicalFeatures
# template_mol_file:'2JKM-ligand-model-bi9.sdf'
# d_upper: 1.5
# d_lower: 0.5
# keep:('Donor','Acceptor','NegIonizable','PosIonizable','Aromatic')
# conformers_num:2
# pList_max_allowed: 30
# failed_allowed:5
# pharmacophore_idx: [0,6,7,8,9]
# disturbance_idxs:6
# disturbance_radius: 1.4
#reward_weight:[0.1,0.25,0.5,0.5]

class Pharmacophore_Disturbance(BaseScoreComponent):
    def __init__(self,parameters: ComponentParameters):
        super(Pharmacophore_Disturbance, self).__init__(parameters)
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        self.d_upper = self.parameters.specific_parameters["d_upper"]
        self.d_lower = self.parameters.specific_parameters["d_lower"]
        self.keep = self.parameters.specific_parameters["keep"]

        self.conformers_num = self.parameters.specific_parameters["conformers_num"]
        self.pList_max_allowed = self.parameters.specific_parameters["pList_max_allowed"]
        self.failed_allowed = self.parameters.specific_parameters["failed_allowed"]
        self.reward_weight = self.parameters.specific_parameters["reward_weight"]

        self.pharmacophore_idxs = self.parameters.specific_parameters["pharmacophore_idxs"]
        self.disturbance_radius = self.parameters.specific_parameters["disturbance_radius"]
        self.disturbance_idxs = self.parameters.specific_parameters["disturbance_idxs"]

        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = fparams

        reference_rawfeats = self.fdef.GetFeaturesForMol(self.template_mol)
        reference_feats = [f for f in reference_rawfeats if f.GetFamily() in self.keep]
        self.reference_fms = FeatMaps.FeatMap(feats = reference_feats,weights=[1]*len(reference_feats),params=self.fmParams)
        self.prob_feats = self.fdef.GetFeaturesForMol(self.template_mol)
        self.prob_points= [list(x.GetPos()) for x in self.prob_feats]

        self.template_contrib = Chem.rdMolDescriptors._CalcCrippenContribs(self.template_mol)
        #self.p4core = self._define_pharmacophore(self.pharmacophore_idxs)

    def _get_disturbance(self):
        x = 255;y=255;z=255
        dist = x*x+y*y+z*z
        while dist>(self.disturbance_radius*self.disturbance_radius):
            x = (random.random()-0.5)*2*self.disturbance_radius
            y = (random.random()-0.5)*2*self.disturbance_radius
            z = (random.random()-0.5)*2*self.disturbance_radius
            dist = x*x+y*y+z*z
        return x,y,z

    def _define_pharmacophore(self):
        required_feats = []
        for idx in self.pharmacophore_idxs:
            if idx >= len(self.prob_feats):
                continue
            if idx == self.disturbance_idxs:
                x_bias, y_bias, z_bias = self._get_disturbance()
                new_x = self.prob_feats[idx].GetPos().x + x_bias
                new_y = self.prob_feats[idx].GetPos().x + y_bias
                new_z = self.prob_feats[idx].GetPos().x + z_bias
                #print('bias:', x_bias, y_bias, z_bias)
                #print('old_pos:')
                #print(self.prob_feats[idx].GetPos().x, self.prob_feats[idx].GetPos().y, self.prob_feats[idx].GetPos().z)

                point = rdGeometry.Point3D(new_x, new_y, new_z)
                new_cf = rdChemicalFeatures.FreeChemicalFeature(self.prob_feats[idx].GetFamily(), point)

                #print('new_pos:')
                #print(new_cf.GetPos().x, new_cf.GetPos().y, new_cf.GetPos().z)
                required_feats.append(new_cf)
            else:
                required_feats.append(self.prob_feats[idx])
        # required_feats = [self.prob_feats[idx] for idx in feat_idxs if idx < len(self.prob_feats)]
        assert len(required_feats) >= 2
        pharm_core = rkcPharmacophore.Pharmacophore(required_feats)
        for idx_i, idx_j in combinations(range(len(required_feats)), 2):
            dist = rdGeometry.Point3D.Distance(required_feats[idx_i].GetPos(), required_feats[idx_j].GetPos())
            pharm_core.setLowerBound(idx_i, idx_j, min(dist - self.d_lower, 0))
            pharm_core.setUpperBound(idx_i, idx_j, dist + self.d_upper)

        self.p4core = pharm_core
        self.ref = [list(p.GetPos()) for p in self.p4core.getFeatures()]

    def calculate_score(self,molecules:List) -> ComponentSummary:
        self._define_pharmacophore()
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
        match, mList = MatchPharmacophoreToMol(input_mol, self.fdef, self.p4core)
        if not match:
            return self.reward_weight[0]
        # round2: pharmacophore distances
        try:
            bounds = rdDistGeom.GetMoleculeBoundsMatrix(input_mol)
            pList = GetAllPharmacophoreMatches(mList, bounds, self.p4core)
        except RuntimeError:
            print("the length pList is ",len(mList))
            return self.reward_weight[0]
        if len(pList) == 0:  # if failed
            return self.reward_weight[1]
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
                    continue
        max_feat_score = 0
        for idx,mol in enumerate(multi_temps1):
            try:
                rawFeats = self.fdef.GetFeaturesForMol(mol)
                featList = [f for f in rawFeats if f.GetFamily() in self.keep]
                score = self.reference_fms.ScoreFeats(featList) / self.reference_fms.GetNumFeatures()
            except RuntimeError:
                print('RuntimeError')
                continue
            if score > max_feat_score:
                max_feat_score = score
        return self.reward_weight[2] + max_feat_score * self.reward_weight[3]

    def get_component_type(self):
        return "pharmacophore_disturbance"
