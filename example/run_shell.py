from src.DLTrainer.pytorch import DLTrainer
from src.DLTrainer.pytorch.trainer_utils import BaseTrainerArgs
from example.model import SimpleConfig, SimpleModel
from example.dataset import SimpleDataset
from example.metrics import calculate_metrics

"""MODELS Dictionary
The models dictionary contains the model classes to be used during training. 
The base format is designed for NLP tasks, however, can be used for non-NLP 
tasks by excluding the TokenizerClass

The following format should be used:

    'nlp_model_name': (ModelConfigClass, ModelClass, DatasetClass, TokenizerClass)
    'non_nlp_model_name': ((ModelConfigClass, ModelClass, DatasetClass)
If 
"""

MODELS = {
    'simple': (SimpleConfig, SimpleModel, SimpleDataset),
}


class TrainerArgs(BaseTrainerArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_size=2
        self.hidden_size=10
        self.output_size=1


args = TrainerArgs(model='simple', do_train=True, logging_steps=1)

trainer = DLTrainer(MODELS, calculate_metrics, args=args)
