import numpy as np
from typing import Iterator, List, Tuple
from pathlib import Path

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl import config
from babybertsrl.word_pieces import wordpiece_tokenize_input
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


class Data:

    def __init__(self,
                 params,
                 train_data_path: Path,
                 dev_data_path: Path,
                 vocab_path_name: str,
                 ):
        """
        loads text from file and puts them in Allen NLP toolkit instances format
        for training with BERT.
        designed to use with CHILDES sentences
        """

        self.params = params
        self.bert_tokenizer = BertTokenizer(vocab_path_name, do_basic_tokenize=False)
        self.lowercase = 'uncased' in config.Data.bert_name

        # load sentences
        self.train_utterances = self.get_utterances_from_file(train_data_path)
        self.dev_utterances = self.get_utterances_from_file(dev_data_path)

        # print info
        print('Found {:,} training sentences ...'.format(self.num_train_sentences))
        print('Found {:,} dev sentences ...'.format(self.num_dev_sentences))
        print()
        for name, sentences in zip(['train', 'dev'],
                                   [self.train_utterances, self.dev_utterances]):
            lengths = [len(s[0]) for s in sentences]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
            print()

        # training with Allen NLP toolkit
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.train_instances = self.make_instances(self.train_utterances)
        self.dev_instances = self.make_instances(self.dev_utterances)

    @property
    def num_train_sentences(self):
        return len(self.train_utterances)

    @property
    def num_dev_sentences(self):
        return len(self.dev_utterances)

    def get_utterances_from_file(self, file_path):
        res = []
        punctuation = {'.', '?', '!'}
        num_skipped = 0
        with file_path.open('r') as f:

            for line in f.readlines():

                tokens = line.strip().split()  # a transcript containing multiple utterances
                if self.lowercase:
                    tokens = [w.lower() for w in tokens]

                # split by utterance marker
                utterances = [[]]
                for w in tokens:
                    utterances[-1].append(w)
                    if w in punctuation:
                        utterances.append([])

                # collect
                for utterance in utterances:

                    # check sentence length
                    if len(utterance) <= config.Data.min_utterance_length:
                        num_skipped += 1
                        continue
                    if len(utterance) > self.params.max_sentence_length:
                        continue

                    res.append(utterance)

        print(f'WARNING: Skipped {num_skipped} utterances which are shorter than {config.Data.min_utterance_length}.')

        return res

    # --------------------------------------------------------- interface with Allen NLP toolkit

    def _text_to_instance(self,
                          lm_in: List[str],
                          lm_mask: List[int],  # masked language modeling
                          lm_tags: List[str] = None) -> Instance:

        # to word-pieces
        word_pieces, offsets, start_offsets = wordpiece_tokenize_input(lm_in,
                                                                       self.bert_tokenizer,
                                                                       self.lowercase)

        # AllenNLP says: In order to override the indexing mechanism, we need to set the `text_id`
        # attribute directly. This causes the indexing to use this id.
        # new_tokens = [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in word_pieces]
        # But, setting text_id causes tokens not to be found by Allen Vocabulary.
        # so, I don't set it:
        new_tokens = [Token(t) for t in word_pieces]
        new_mask = convert_lm_mask_to_wordpiece_lm_mask(lm_mask, offsets)

        # meta data only has whole words
        metadata_dict = dict()
        metadata_dict['offsets'] = start_offsets
        metadata_dict['lm_in'] = lm_in
        metadata_dict['masked_indices'] = lm_mask  # mask is list containing zeros and ones
        metadata_dict['gold_lm_tags'] = lm_tags  # is just a copy of the input without the mask

        text_field = TextField(new_tokens, self.token_indexers)
        fields = {'tokens': text_field,
                  'mask_indicator': SequenceLabelField(new_mask, text_field),
                  'lm_tags': SequenceLabelField(word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, utterances) -> Iterator[Instance]:
        """
        because lazy is by default False, return a list rather than a generator.
        When lazy=False, the generator would be converted to a list anyway.

        roughly equivalent to Allen NLP toolkit dataset.read()

        """
        res = []
        for utterance in utterances:

            # collect each multiple times, each time with a different masked word
            utterance_length = len(utterance)
            num_masked = min(utterance_length, self.params.num_masked)
            for masked_id in range(num_masked):

                # collect instance
                lm_in, lm_mask, lm_tags = prepare_utterance_for_instance(utterance, masked_id)
                instance = self._text_to_instance(lm_in, lm_mask, lm_tags)
                res.append(instance)

        return res  # TODO how to return a generator here? instances are fed to vocab which require list