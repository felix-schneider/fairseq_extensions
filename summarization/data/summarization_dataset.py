import numpy as np
import torch

from fairseq.data import FairseqDataset, language_pair_dataset


class SummarizationDataset(FairseqDataset):
    def __init__(
            self, src, src_sizes, dictionary,
            tgt=None, tgt_sizes=None,
            speakers=None,
            input_feeding=True,
            left_pad_source=False, left_pad_target=False,
            strip_bos_from_source=True
    ):
        self.src = src
        self.tgt = tgt
        self.speakers = speakers
        src_sizes = np.array(src_sizes)
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.dictionary = dictionary
        self.input_feeding = input_feeding
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.strip_bos_from_source = strip_bos_from_source

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index] if self.tgt is not None else None

        if self.tgt is not None and tgt_item[-1] != self.dictionary.eos_index:
            tgt_item = torch.cat([tgt_item, torch.LongTensor([self.dictionary.eos_index])])

        if self.strip_bos_from_source and src_item[0] == self.dictionary.bos_index:
            src_item = src_item[1:]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item
        }
        return example

    def collater(self, samples):
        return language_pair_dataset.collate(
            samples, self.dictionary.pad_index, self.dictionary.eos_index,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding
        )

    def num_tokens(self, index):
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 1

    def size(self, index):
        return self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)

    def ordered_indices(self):
        indices = np.argsort(self.src_sizes, kind="mergesort")
        # if self.tgt_sizes is not None:
        #     indices = np.argsort(self.tgt_sizes[indices], kind="mergesort")
        return indices
