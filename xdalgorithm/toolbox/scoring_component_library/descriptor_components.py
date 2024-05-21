from typing import List
from rdkit.Chem.Descriptors import MolWt
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.scoring_component_library.physchem.base_physchem_component import BasePhysChemComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from rdkit.Chem.Descriptors import TPSA 
from rdkit.Chem import Lipinski

class MolWeight(BasePhysChemComponent):
	def __init__(self, parameters: ComponentParameters):
		super().__init__(parameters)

	def calculate_score(self,molecules:List) -> ComponentSummary:
		scores = self._score_molecules(molecules)
		score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
		return score_summary

	def _score_molecules(self,molecules): 
		return np.array([self._score_molecule(molecule) for molecule in molecules])
	
	def _score_molecule(self,input_mol):
		return MolWt(input_mol)

	def get_component_type(self):
		return "molecular_weight"

class TPSA(BasePhysChemComponent):
        def __init__(self, parameters: ComponentParameters):
                super().__init__(parameters)

        def calculate_score(self,molecules:List) -> ComponentSummary:
                scores = self._score_molecules(molecules)
                score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
                return score_summary

        def _score_molecules(self,molecules):
                return np.array([self._score_molecule(molecule) for molecule in molecules])

        def _score_molecule(self,input_mol):
                return TPSA(input_mol)

        def get_component_type(self):
                return "tpsa"

class NumRings(BasePhysChemComponent):
        def __init__(self, parameters: ComponentParameters):
                super().__init__(parameters)

        def calculate_score(self,molecules:List) -> ComponentSummary:
                scores = self._score_molecules(molecules)
                score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
                return score_summary

        def _score_molecules(self,molecules):
                return np.array([self._score_molecule(molecule) for molecule in molecules])

        def _score_molecule(self,input_mol):
                return Lipinski.NumAliphaticRings(input_mol)+Lipinski.NumAromaticRings(input_mol)

        def get_component_type(self):
                return "num_rings"


class HBA(BasePhysChemComponent):
        def __init__(self, parameters: ComponentParameters):
                super().__init__(parameters)

        def calculate_score(self,molecules:List) -> ComponentSummary:
                scores = self._score_molecules(molecules)
                score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
                return score_summary

        def _score_molecules(self,molecules):
                return np.array([self._score_molecule(molecule) for molecule in molecules])

        def _score_molecule(self,input_mol):
                return  Lipinski.NumHAcceptors(input_mol)

        def get_component_type(self):
                return "num_hba_lipinski"

class HBD(BasePhysChemComponent):
        def __init__(self, parameters: ComponentParameters):
                super().__init__(parameters)

        def calculate_score(self,molecules:List) -> ComponentSummary:
                scores = self._score_molecules(molecules)
                score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
                return score_summary

        def _score_molecules(self,molecules):
                return np.array([self._score_molecule(molecule) for molecule in molecules])

        def _score_molecule(self,input_mol):
                return Lipinski.NumHDonors(input_mol)

        def get_component_type(self):
                return "num_hbd_lipinski"


class NumRotatableBonds(BasePhysChemComponent):
        def __init__(self, parameters: ComponentParameters):
                super().__init__(parameters)

        def calculate_score(self,molecules:List) -> ComponentSummary:
                scores = self._score_molecules(molecules)
                score_summary = ComponentSummary(total_score=scores,parameters = self.parameters)
                return score_summary

        def _score_molecules(self,molecules):
                return np.array([self._score_molecule(molecule) for molecule in molecules])

        def _score_molecule(self,input_mol):
                return Lipinski.NumRotableBons(input_mol)

        def get_component_type(self):
                return "num_rotatable_bonds"

