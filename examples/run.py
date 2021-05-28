from model import SimpleConfig, SimpleModel
from dataset import SimpleDataset
from metrics import calculate_metrics
import sys
sys.path.append('/Users/nickvasko/projects/DLTrainer')
from src.DLTrainer.pytorch import DLTrainer

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


def additional_arg_parser(parser):
    """Custom parameters can be passed through the command line by creating a custom
    argument parsing function.

    :param parser: (argparse.ArgumentParser)
    :return: parser: (argparse.ArgumentParser)
    """
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--output_size', type=int, default=1)
    return parser


if __name__ == "__main__":
    trainer = DLTrainer(MODELS, calculate_metrics, additional_arg_parser)
