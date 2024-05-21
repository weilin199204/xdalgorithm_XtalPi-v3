from typing import List

import numpy as np
from rdkit import Chem

from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary

import os
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
#from py3Dmol import view
from IPython.display import display
import copy
from collections import defaultdict
from itertools import combinations

from rdkit.Chem import rdMolAlign
# template_mol_file:'2JKM-ligand-model-bi9.sdf'
# d_upper: 1.5
# d_lower: 0.5
# keep:('Donor','Acceptor','NegIonizable','PosIonizable','Aromatic')
# pharmacophore_idx: (0,6,7,8,9)

#
#class ComponentParameters:
#    component_type: str
#    name: str
#    weight: float
#    smiles: List[str]
#    model_path: str
#    specific_parameters: dict = None

# round0: valid check
# round1: pharmacophore matched
# round2: pharmacophore distance
# round3: ScoreFeats, ScoreFeats
#reward_weight:(0.1,0.25,0.5,0.5)

class Pharmacophore(BaseScoreComponent):
    def __init__(self,parameters: ComponentParameters):
        super(Pharmacophore, self).__init__(parameters)
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        self.d_upper = self.parameters.specific_parameters["d_upper"]
        self.d_lower = self.parameters.specific_parameters["d_lower"]
        self.keep = self.parameters.specific_parameters["keep"]
        self.pharmacophore_idxs = self.parameters.specific_parameters["pharmacophore_idxs"]
        self.reward_weight = self.parameters.specific_parameters["reward_weight"]

        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = fparams

        reference_rawfeats = self.fdef.GetFeaturesForMol(self.template_mol)
        reference_feats = [f for f in reference_rawfeats if f.GetFamily() in self.keep]
        self.reference_fms = FeatMaps.FeatMap(feats = reference_feats,weights=[1]*len(reference_feats),params=self.fmParams)
        self.prob_feats = self.fdef.GetFeaturesForMol(self.template_mol)
        self.prob_points= [list(x.GetPos()) for x in self.prob_feats]

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
        match, mList = MatchPharmacophoreToMol(input_mol, self.fdef, self.p4core)
        if not match:
            return self.reward_weight[0]
        # round2: pharmacophore distances
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(input_mol)
        pList = GetAllPharmacophoreMatches(mList, bounds, self.p4core)
        if len(pList) == 0:  # if failed
            return self.reward_weight[1]
        # round3:
        ps = Chem.AllChem.ETKDG()
        ps.randomSeed = 0xf00d
        Chem.AllChem.EmbedMolecule(input_mol, ps)
        # try:
        try:
            o3d = rdMolAlign.GetO3A(input_mol, self.template_mol)
            o3d.Align()
        except ValueError:
            return 0.0
        # except ValueError:
        #    return 0.1
        rawFeats = self.fdef.GetFeaturesForMol(input_mol)
        featList = [f for f in rawFeats if f.GetFamily() in self.keep]
        # fm = FeatMaps.FeatMap(feats=featList, weights=[1] * len(featList), params=self.fmParams)
        score = self.reference_fms.ScoreFeats(featList) / self.reference_fms.GetNumFeatures()
        return self.reward_weight[2] + score * self.reward_weight[3]

    def get_component_type(self):
        return "pharmacophore"
