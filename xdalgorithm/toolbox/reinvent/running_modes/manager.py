import xdalgorithm.toolbox.reinvent.utils.general as utils_general
from xdalgorithm.toolbox.reinvent.models.model import Model
from xdalgorithm.toolbox.reinvent.running_modes.configurations import TransferLearningConfiguration, ScoringRunnerConfiguration, \
    ReinforcementLearningConfiguration, SampleFromModelConfiguration, CreateModelConfiguration, \
    AdaptiveLearningRateConfiguration, InceptionConfiguration, ReinforcementLearningComponents, \
    GeneralConfigurationEnvelope, ScoringRunnerComponents
from xdalgorithm.toolbox.reinvent.running_modes.create_model.create_model import CreateModelRunner
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.inception import Inception
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.reinforcement_runner import ReinforcementRunner
from xdalgorithm.toolbox.reinvent.running_modes.sampling.sample_from_model import SampleFromModelRunner
from xdalgorithm.toolbox.reinvent.running_modes.scoring.scoring_runner import ScoringRunner
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.transfer_learning_runner import TransferLearningRunner
from xdalgorithm.toolbox.reinvent.running_modes.validation.validation_runner import ValidationRunner
from xdalgorithm.toolbox.reinvent.scaffold.scaffold_filter_factory import ScaffoldFilterFactory
from xdalgorithm.toolbox.reinvent.scaffold.scaffold_parameters import ScaffoldParameters
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.scoring_function_factory import ScoringFunctionFactory
from xdalgorithm.toolbox.reinvent.scoring.scoring_function_parameters import ScoringFuncionParameters # name,parameters,parallel
from xdalgorithm.toolbox.reinvent.utils.enums.running_mode_enum import RunningModeEnum

class Manager:

    def __init__(self, configuration):
        self.running_mode_enum = RunningModeEnum()
        self.configuration = GeneralConfigurationEnvelope(**configuration)
        #self.start_smiles = self.configuration.parameters.get('start_smiles', '')
        utils_general.set_default_device_cuda()

    def _run_create_empty_model(self):
        config = CreateModelConfiguration(**self.configuration.parameters)
        runner = CreateModelRunner(self.configuration, config)
        runner.run()

    def _run_transfer_learning(self):
        config = TransferLearningConfiguration(**self.configuration.parameters)
        model = Model.load_from_file(config.input_model_path)
        adaptive_lr_config = AdaptiveLearningRateConfiguration(**config.adaptive_lr_config)
        adaptive_learning_rate = AdaptiveLearningRate(model, self.configuration, adaptive_lr_config)
        runner = TransferLearningRunner(model, config, adaptive_learning_rate)
        runner.run()

    def _run_reinforcement_learning(self):
        rl_components = ReinforcementLearningComponents(**self.configuration.parameters) #
        # reinforcement_learning: dict
        # scoring_function: dict
        # diversity_filter: dict
        # inception: dict
        # start_smiles: str = ''
        scaffold_filter = self._setup_scaffold_filter(rl_components.diversity_filter)
        scoring_function = self._setup_scoring_function(rl_components.scoring_function)
        rl_config = ReinforcementLearningConfiguration(**rl_components.reinforcement_learning)
        inception_config = InceptionConfiguration(**rl_components.inception)
        inception = Inception(inception_config, scoring_function, Model.load_from_file(rl_config.prior))
        start_smiles = rl_components.start_smiles
        runner = ReinforcementRunner(
            self.configuration, rl_config, scaffold_filter, scoring_function,
            inception=inception,
            start_smiles=start_smiles
        )
        runner.run()

    def _setup_scaffold_filter(self, scaffold_parameters):
        scaffold_parameters = ScaffoldParameters(**scaffold_parameters) # name, minscore, nbmax, minsimilarity
        scaffold_factory = ScaffoldFilterFactory()
        scaffold = scaffold_factory.load_scaffold_filter(scaffold_parameters)
        return scaffold

    def _setup_scoring_function(self, scoring_function_parameters):
        scoring_function_parameters = ScoringFuncionParameters(**scoring_function_parameters)
        # return a scoring function instance
        scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        return scoring_function_instance

    def _run_sampling(self):
        config = SampleFromModelConfiguration(**self.configuration.parameters)
        runner = SampleFromModelRunner(self.configuration, config)
        runner.run()

    def _run_scoring(self):
        sr_components = ScoringRunnerComponents(**self.configuration.parameters) #scoring,scoring_function
        # print(sr_components.scoring_function)
        scoring_function = self._setup_scoring_function(sr_components.scoring_function)
        scoring_config = ScoringRunnerConfiguration(**sr_components.scoring)
        runner = ScoringRunner(configuration=self.configuration,
                               config=scoring_config,
                               scoring_function=scoring_function)
        runner.run()

    def _run_validation(self):
        config = ComponentParameters(**self.configuration.parameters)
        runner = ValidationRunner(self.configuration, config)
        runner.run()

    def run(self):
        """determines from the configuration object which type of run it is expected to start"""
        switcher = {
            self.running_mode_enum.TRANSFER_LEARNING: self._run_transfer_learning,
            self.running_mode_enum.REINFORCEMENT_LEARNING: self._run_reinforcement_learning,
            self.running_mode_enum.SAMPLING: self._run_sampling,
            self.running_mode_enum.SCORING: self._run_scoring,
            self.running_mode_enum.CREATE_MODEL: self._run_create_empty_model,
            self.running_mode_enum.VALIDATION: self._run_validation
        }
        job = switcher.get(self.configuration.run_type, lambda: TypeError)
        job()
