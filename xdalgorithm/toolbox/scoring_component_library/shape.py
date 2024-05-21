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


import sys
import os
#sys.path.append('../pyshapeit/')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import RDConfig


class Shape(BaseScoreComponent):
    def __init__(self,parameters: ComponentParameters):
        super(Shape, self).__init__(parameters):
        self.ref_sdf = self.parameters.specific_parameters["ref_sdf"]
        self.ref_mol = Chem.SDMolSupplier(self.ref_sdf, removeHs=False)[0]
	self.max_confshapes=self.parameters.specific_parameters["max_confshape"]
	self.num_threads=num_threads
	self.use_chemaxon=use_chemaxon
	self.protonation=protonation

    def calculate_score(self,molecules:List) -> ComponentSummary:
        scores = self._score_molecules(molecules)
        score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
        return score_summary


    def _score_molecules(self,molecules):
        return np.array([self._score_molecule(molecule) for molecule in molecules])

    def _score_molecule(self,input_mol):

            try:
               #shape_score  = cpyshapeit.AlignMol(self.template_mol, mol)
		shape_score = 1.0
            except ValueError:
                shape_score = 0.0
            shape_scores.append(shape_score)
        return np.array(shape_scores, dtype=np.float32)



    def get_component_type(self):
        return "shape"


