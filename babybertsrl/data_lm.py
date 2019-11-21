import numpy as np
from typing import Iterator, List, Tuple
from pathlib import Path
import pyprind

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl import config
from babybertsrl.word_pieces import wordpiece_tokenize
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
        self.lowercase = False  # set to false because [MASK] must be uppercase?
        self.bert_tokenizer = BertTokenizer(vocab_path_name,
                                            do_basic_tokenize=False,
                                            do_lower_case=self.lowercase)

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

        # use Allen NLP logic:
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}  # specifies how a token is indexed
        self.train_instances = self.make_instances(self.train_utterances)
        self.dev_instances = self.make_instances(self.dev_utterances)

        # use vocab to store labels vocab, input vocab is stored in bert_tokenizer.vocab
        # what from_instances() does:
        # 1. it iterates over all instances, and all fields, and all toke indexers
        # 2. the token indexer is used to update vocabulary count, skipping words whose text_id is already set
        self.vocab = Vocabulary.from_instances(self.train_instances + self.dev_instances)
        self.vocab.print_statistics()

    @property
    def num_train_sentences(self):
        return len(self.train_utterances)

    @property
    def num_dev_sentences(self):
        return len(self.dev_utterances)

    def get_utterances_from_file(self, file_path):
        res = []
        punctuation = {'.', '?', '!'}
        num_too_small = 0
        num_too_large = 0
        with file_path.open('r') as f:

            for line in f.readlines():

                # tokenize transcript
                transcript = line.strip().split()  # a transcript containing multiple utterances
                if self.lowercase:
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
        lm_in_word_pieces, offsets, start_offsets = wordpiece_tokenize(lm_in,
                                                                       self.bert_tokenizer,
                                                                       self.lowercase)
        lm_tags_word_pieces, offsets, start_offsets = wordpiece_tokenize(lm_tags,
                                                                         self.bert_tokenizer,
                                                                         self.lowercase)

        # meta data only has whole words
        metadata_dict = dict()
        metadata_dict['offsets'] = start_offsets
        metadata_dict['lm_in'] = lm_in
        metadata_dict['masked_indices'] = lm_mask  # mask is list containing zeros and ones
        metadata_dict['gold_lm_tags'] = lm_tags  # is just a copy of the input without the mask

        # fields
        tokens = [Token(t) for t in lm_in_word_pieces]
        text_field = TextField(tokens, self.token_indexers)
        new_mask = convert_lm_mask_to_wordpiece_lm_mask(lm_mask, offsets)
        fields = {'tokens': text_field,
                  'mask_indicator': SequenceLabelField(new_mask, text_field),
                  'lm_tags': SequenceLabelField(lm_tags_word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, utterances) -> Iterator[Instance]:
        """
        because lazy is by default False, return a list rather than a generator.
        When lazy=False, the generator would be converted to a list anyway.

        roughly equivalent to Allen NLP toolkit dataset.read()

        """
        res = []
        progress = pyprind.ProgBar(len(utterances), stream=2, title='Making instances')
        for utterance in utterances:

            # collect each multiple times, each time with a different masked word
            utterance_length = len(utterance)
            num_masked = min(utterance_length, self.params.num_masked)
            for masked_id in range(num_masked):

                # collect instance
                lm_in, lm_mask, lm_tags = prepare_utterance_for_instance(utterance, masked_id)
                instance = self._text_to_instance(lm_in, lm_mask, lm_tags)
                res.append(instance)

            progress.update()

        return res  # TODO how to return a generator here? instances are fed to vocab which require list