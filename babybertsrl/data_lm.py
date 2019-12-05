import numpy as np
from typing import Iterator, List, Tuple, Optional
from pathlib import Path

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl import config
from babybertsrl.word_pieces import wordpiece
from babybertsrl.word_pieces import convert_lm_mask_to_wordpiece_lm_mask


def prepare_utterance_for_instance(words: List[str],
                                   masked_id: int,
                                   ) -> Tuple[List[str], List[int], List[str]]:

    lm_in = ['[MASK]' if i == masked_id else words[i] for i in range(len(words))]
    lm_mask = [1 if i == masked_id else 0 for i in range(len(words))]
    lm_tags = words

    if all([x == 0 for x in lm_mask]):
        raise ValueError('Mask indicator contains zeros only. ')

    return lm_in, lm_mask, lm_tags


class DataLM:

    def __init__(self,
                 params,
                 data_path: Path,
                 bert_tokenizer: Optional[BertTokenizer] = None,
                 ):
        """
        loads text from file and puts them in Allen NLP toolkit instances format
        for training with BERT.
        designed to use with CHILDES sentences

        """

        self.params = params

        # load utterances
        self.utterances = self.get_utterances_from_file(data_path)
        lengths = [len(s[0]) for s in self.utterances]
        print('Found {:,} utterances'.format(len(self.utterances)))
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')
        print()

        if not bert_tokenizer:  # e.g. when only using utterances for srl tagging
            return

        self.bert_tokenizer = bert_tokenizer

        # instances
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}  # specifies how a token is indexed
        self.instances = self.make_instances(self.utterances)

    def get_utterances_from_file(self, file_path):
        res = []
        punctuation = {'.', '?', '!'}
        num_too_small = 0
        num_too_large = 0
        with file_path.open('r') as f:

            for line in f.readlines():

                # tokenize transcript
                transcript = line.strip().split()  # a transcript containing multiple utterances
                transcript = [w.lower() for w in transcript]

                # split transcript into utterances
                utterances = [[]]
                for w in transcript:
                    utterances[-1].append(w)
                    if w in punctuation:
                        utterances.append([])

                # collect utterances
                for utterance in utterances:

                    # check  length
                    if len(utterance) <= config.Data.min_utterance_length:
                        num_too_small += 1
                        continue
                    if len(utterance) > self.params.max_sentence_length:
                        num_too_large += 1
                        continue

                    res.append(utterance)

        print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {config.Data.min_utterance_length}.')
        print(f'WARNING: Skipped {num_too_large} utterances which are larger than {self.params.max_sentence_length}.')

        return res

    # --------------------------------------------------------- interface with Allen NLP toolkit

    def _text_to_instance(self,
                          lm_in: List[str],
                          lm_mask: List[int],
                          lm_tags: List[str],
                          ) -> Instance:

        # to word-pieces
        lm_in_word_pieces, offsets, start_offsets = wordpiece(lm_in,
                                                              self.bert_tokenizer,
                                                              lowercase_input=False)
        lm_tags_word_pieces, _, _ = wordpiece(lm_tags,
                                              self.bert_tokenizer,
                                              lowercase_input=False)
        lm_mask_word_pieces = convert_lm_mask_to_wordpiece_lm_mask(lm_mask, offsets)

        # meta data only has whole words
        metadata_dict = dict()
        metadata_dict['offsets'] = start_offsets
        metadata_dict['lm_in'] = lm_in
        metadata_dict['masked_indices'] = lm_mask  # mask is list containing zeros and ones
        metadata_dict['gold_lm_tags'] = lm_tags  # is just a copy of the input without the mask

        # fields
        tokens = [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in lm_in_word_pieces]
        text_field = TextField(tokens, self.token_indexers)

        if len(lm_in_word_pieces) != len(lm_tags_word_pieces):
            # the code does not yet support custom word-pieces in vocabulary,
            # because it does not handle case when masked word is split into word-pieces.
            # In such a case, input and output length are mismatched.
            # The output is longer, because it contains more than 1 piece for the masked whole word in the input.
            raise UserWarning('A word-piece split word was masked. Word pieces are not supported')

        fields = {'tokens': text_field,
                  'mask_indicator': SequenceLabelField(lm_mask_word_pieces, text_field),
                  'lm_tags': SequenceLabelField(lm_tags_word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, utterances) -> Iterator[Instance]:
        """
        roughly equivalent to Allen NLP toolkit dataset.read()

        """
        for utterance in utterances:

            # collect each multiple times, each time with a different masked word
            utterance_length = len(utterance)
            num_masked = min(utterance_length, self.params.num_masked)
            for masked_id in range(num_masked):

                # collect instance
                lm_in, lm_mask, lm_tags = prepare_utterance_for_instance(utterance, masked_id)
                instance = self._text_to_instance(lm_in, lm_mask, lm_tags)

                yield instance