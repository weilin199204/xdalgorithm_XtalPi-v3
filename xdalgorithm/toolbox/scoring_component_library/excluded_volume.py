import mdtraj as md
from rdkit.Chem import ChemicalFeatures
import os
import copy
import numpy as np
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import RDConfig
from rdkit.Chem import rdGeometry
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Numerics import rdAlignment

#from collections import defaultdict
#from itertools import combinations

# template_mol_file:'2JKM-ligand-model-bi9.sdf',
# lig_name:"BII",
# pdb_name: "2JKM.pdb",
# atom_width: 1.5,
# penalty_weight: -3
#
#class ComponentParameters:
#    component_type: str
#    name: str
#    weight: float
#    smiles: List[str]
#    model_path: str
#    specific_parameters: dict = None

from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary


def getExcludedVolumn(pdb, lig_name):
    p = md.load_pdb(pdb)
    lig = p.topology.select('resname "%s"' %lig_name)
    protein_limit = p.topology.select('protein')[-1]
    neighbors = md.compute_neighbors(p, 0.5, lig, periodic=False)[0]
    excluded_coords = []
    s = set()
    for n in neighbors:
        a = p.topology.atom(n)
        r = a.residue
        if n > protein_limit:
            break
        if not r.resSeq in s:
            s.add(r.resSeq)
            excluded_coords.append([float(x) * 10 for x in p.xyz[0][next(r.atoms_by_name('C')).index]])
            excluded_coords.append([float(x) * 10 for x in p.xyz[0][next(r.atoms_by_name('CA')).index]])
            excluded_coords.append([float(x) * 10 for x in p.xyz[0][next(r.atoms_by_name('O')).index]])
            excluded_coords.append([float(x) * 10 for x in p.xyz[0][next(r.atoms_by_name('N')).index]])
            if not r.name == 'GLY':
                c = next(r.atoms_by_name('CB'))
                excluded_coords.append([float(x) * 10 for x in p.xyz[0][c.index]])
    return excluded_coords

class ExcludedVolume(BaseScoreComponent):
    def __init__(self,parameters: ComponentParameters):
        super(Excluded_Volumn, self).__init__(parameters)
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        self.LIGNAME = self.parameters.specific_parameters["lig_name"]
        self.PDBNAME = self.parameters.specific_parameters["pdb_name"]
        self.atom_width = self.parameters.specific_parameters["atom_width"]
        self.penalty_weight = self.parameters.specific_parameters["penalty_weight"]
        assert  self.penalty_weight > 0.0

        import os
        defined_fdef_filepath = os.path.join(os.path.dirname(__file__), 'defined_BaseFeatures.fdef')
        self.defined_fdef = AllChem.BuildFeatureFactory(defined_fdef_filepath)

        exclude = getExcludedVolumn(self.PDBNAME,self.LIGNAME)

        p4core_score = [ChemicalFeatures.FreeChemicalFeature('Any',
                        Geometry.Point3D(float(ex[0]), float(ex[1]), float(ex[2])))
                        for ex in exclude]
        p4core_weights = [-self.penalty_weight] * len(exclude)
        fmParams = {}
        for k in self.defined_fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            fmParams[k] = fparams
        fmParams['Any'].width = self.atom_width
        self.p4map_score = FeatMaps.FeatMap(feats=p4core_score, weights=p4core_weights, params=fmParams)


    def calculate_score(self,molecules:List) -> ComponentSummary:
        scores = self._score_molecules(molecules)
        score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
        return score_summary

    def _score_molecules(self,molecules):
        #molecules:a list containing valid Mol objects
        #return an array scores
        return np.array([self._score_molecule(molecule) for molecule in molecules])

    def _score_molecule(self,input_mol):
        #input_mol: a Mol object
        # Align the conformation
        ps = AllChem.ETKDG()
        ps.randomSeed = 0xf00d
        AllChem.EmbedMolecule(input_mol,ps)
        try:
            o3d = rdMolAlign.GetO3A(input_mol,self.template_mol)
            o3d.Align()
        except ValueError:
            return -100 

        rawFeats = self.defined_fdef.GetFeaturesForMol(input_mol)
        featList = [f for f in rawFeats]
        score = self.p4map_score.ScoreFeats(featList) / self.p4map_score.GetNumFeatures()
        #return a weight
        return score

    def get_component_type(self):
        return "excluded_volumn"
