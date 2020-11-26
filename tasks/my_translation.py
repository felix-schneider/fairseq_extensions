from fairseq.data import LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task("my_translation")
class MyTranslationTask(TranslationTask):
    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            left_pad_source=self.args.left_pad_source,
            constraints=constraints
        )
