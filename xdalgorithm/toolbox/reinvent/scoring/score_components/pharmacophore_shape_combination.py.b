from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from rdkit import Chem
from xdalgorithm.toolbox.scoring_component_library.pharmacophore_shape_combination \
    import PharmacophoreShapeCombination as PharmacophoreShapeCombinationScorer


class PharmacophoreShapeCombination(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super(PharmacophoreShapeCombination,self).__init__(parameters)
        self.conformers_num = self.parameters.specific_parameters["conformers_num"]
        self.failed_allowed = self.parameters.specific_parameters["max_to_try"]
        self.reward_weight = self.parameters.specific_parameters["reward_weights"]
        self.d_upper = self.parameters.specific_parameters["d_upper"]
        self.d_lower = self.parameters.specific_parameters["d_lower"]
        self.keep = self.parameters.specific_parameters["valid_p4cores"]
        self.pList_max_allowed = self.parameters.specific_parameters["max_num_p4cores"]
        self.pharmacophore_idxs = self.parameters.specific_parameters["pharmacophore_idxs"]
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        self.use_chemaxon = self.parameters.specific_parameters.get('use_chemaxon', False)
        self.protonation = self.parameters.specific_parameters.get('protonation', True)
        if "pharmacophore_weights" in self.parameters.specific_parameters:
            self.pharmacophore_weights = self.parameters.specific_parameters["pharmacophore_weights"]
        else:
            self.pharmacophore_weights = [1.] * len(self.pharmacophore_idxs)


        self.scorer = PharmacophoreShapeCombinationScorer(
            conformers_num=self.conformers_num,
            failed_allowed=self.failed_allowed,
            reward_weight=self.reward_weight,
            d_upper=self.d_upper,
            d_lower=self.d_lower,
            keep=self.keep,
            pList_max_allowed=self.pList_max_allowed,
            pharmacophore_idxs=self.pharmacophore_idxs,
            template_mol=self.template_mol,
            use_chemaxon=self.use_chemaxon,
            protonation=self.protonation,
            pharmacophore_weights=self.pharmacophore_weights 
        )

    def calculate_score(self, molecules: List) -> ComponentSummary:
        scores = [self.scorer.calculate_score(molecule) for molecule in molecules]
        print(scores)
        score_summary = ComponentSummary(total_score=scores, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "pharmacophore_shape_combination"
