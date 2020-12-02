import logging
import os
from typing import List

import numpy as np
import torch

from fairseq.data import indexed_dataset, data_utils
from fairseq.tasks import register_task, FairseqTask
from ..data.summarization_dataset import SummarizationDataset

logger = logging.getLogger(__name__)


@register_task("summarization")
class SummarizationTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('data', help='data directory')
        parser.add_argument('--max-source-positions', default=35536, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=2048, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--left-pad-source', action="store_true")
        parser.add_argument('--left-pad-target', action="store_true")
        parser.add_argument('--multitask', action="store_true")

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.load_dictionary(os.path.join(args.data, "dict.txt"))
        logger.info(f'Dictionary: {len(dictionary)} types')

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        self.args.left_pad_source = getattr(self.args, "left_pad_source", False)
        self.args.left_pad_target = getattr(self.args, "left_pad_target", False)

        src_filename = os.path.join(self.args.data, f"{split}.fulltext")
        tgt_filename = os.path.join(self.args.data, f"{split}.summary")

        if not indexed_dataset.dataset_exists(src_filename, impl=self.args.dataset_impl):
            raise FileNotFoundError(f"Dataset not found: {split} ({src_filename})")
        src_dataset = data_utils.load_indexed_dataset(src_filename, self.dictionary, self.args.dataset_impl)
        tgt_dataset = data_utils.load_indexed_dataset(tgt_filename, self.dictionary, self.args.dataset_impl)

        logger.info(f"{src_filename} {split} fulltext {len(src_dataset)} lines")
        if tgt_dataset is not None:
            logger.info(f"{tgt_filename} {split} fulltext {len(tgt_dataset)} lines")

        # TODO: include tagging
        self.datasets[split] = SummarizationDataset(
            src_dataset, src_dataset.sizes, self.dictionary,
            tgt_dataset, tgt_dataset.sizes if tgt_dataset is not None else None,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
        )

    def build_dataset_for_inference(self, src_tokens: List[torch.Tensor], src_lengths: List[int],
                                    **kwargs) -> torch.utils.data.Dataset:
        return SummarizationDataset(
            src_tokens, src_lengths, self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
