from . import trainer_utils

import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter


class DLTrainer:
    """
    Base trainer class for training deep learning models.
    """
    def __init__(self, MODELS, additional_arg_parser=None):
        """Constructor

        :param MODELS:
        :param additional_arg_parser:
        """
        args, logger = trainer_utils.train_setup(additional_arg_parser)

        # Barrier to make sure only the first process in distributed training downloads model & vocab
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        model_checkpoint = os.path.join(args.save_dir, 'checkpoint') if args.pretrained_checkpoint == '' \
            else args.pretrained_checkpoint

        classes = MODELS[args.model_type]
        if len(classes) == 4:
            config_class, model_class, dataset_class, tokenizer_class = classes
        elif len(classes) == 3:
            config_class, model_class, dataset_class = classes
        else:
            raise ValueError(f"MODELS class list must contain either 4 elements for NLP or 3 elements otherwise. "
                             "Received {len(classes)} elements.")
        if args.load_pretrained:
            config = config_class.from_pretrained(model_checkpoint)
            model = model_class.from_pretrained(model_checkpoint)
            if len(classes) == 4:
                tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
        else:
            config = config_class(args)
            model = model_class(config)
            if len(classes) == 4:
                tokenizer = tokenizer_class(config)

        num_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model has a total of {num_params} trainable parameters.")

        # End of distributed training barrier
        if args.local_rank == 0:
            torch.distributed.barrier()

        logger.info("Training/evaluation parameters %s", args)

        # Training
        if args.do_train:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            # train_dataset
            # train_dataloader
            val_dataloader = None
            if not args.no_eval_during_training:
                pass
                # val_dataset
                # val_dataloader

            # train

        if args.do_eval:
            pass
            # val_dataset
            # val_dataloader

        if args.do_test:
            pass
            # test_dataset
            # test_dataloader

    def _default_optimizer(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def save_model(self):
        """Save a checkpoint in the output directory"""
        if os.path.isfile(self.checkpoint_path):
            return
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        state_dict = model_to_save.state_dict()
        model_output_file = self.checkpoint_path + '/pytorch_model.bin'
        torch.save(state_dict, model_output_file)



