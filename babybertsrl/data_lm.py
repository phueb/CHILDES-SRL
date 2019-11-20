import numpy as np
from typing import Iterator, List, Dict, Any
from pathlib import Path
import random

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl import config
from babybertsrl.word_pieces import wordpiece_tokenize_input
from babybertsrl.word_pieces import convert_lm_mask_to_wordpiece_lm_mask
from babybertsrl.word_pieces import convert_lm_tags_to_wordpiece_lm_tags


class Data:

    def __init__(self,
                 params,
                 train_data_path: Path,
                 dev_data_path: Path,
                 ):
        """
        loads text from file and puts them in Allen NLP toolkit instances format
        for training with BERT.
        designed to use with CHILDES sentences
        """

        self.params = params

        # TODO the tokenizer loads word-pieces from the uncased model;
        # TODO this is potentially problematic, because i am not using the uncased model
        # TODO instead, i am using a custom vocabulary
        # TODO should i make a custom vocab file to build a custom bert_tokenizer?

        self.bert_tokenizer = BertTokenizer.from_pretrained(config.Data.bert_name)
        self.lowercase = 'uncased' in config.Data.bert_name

        # load sentences
        self.train_sentences = self.get_sentences_from_file(train_data_path)
        self.dev_sentences = self.get_sentences_from_file(dev_data_path)

        # print info
        print('Found {:,} training sentences ...'.format(self.num_train_sentences))
        print('Found {:,} dev sentences ...'.format(self.num_dev_sentences))
        print()
        for name, sentences in zip(['train', 'dev'],
                                      [self.train_sentences, self.dev_sentences]):
            lengths = [len(s[0]) for s in sentences]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
            print()

        # training with Allen NLP toolkit
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.train_instances = self.make_instances(self.train_sentences)
        self.dev_instances = self.make_instances(self.dev_sentences)

    @property
    def num_train_sentences(self):
        return len(self.train_sentences)

    @property
    def num_dev_sentences(self):
        return len(self.dev_sentences)

    def get_sentences_from_file(self, file_path,
                                num_masked_per_sentence: int = 1):  # TODO should be in params
        """
        Read tokenized sentences from file.
          Return:
            A list with elements of structure [[words], [lm_tags]]
        """
        sentences = []
        punctuation = {'.', '?', '!'}
        num_skipped = 0
        with file_path.open('r') as f:

            for line in f.readlines():

                tokens = line.strip().split()  # a transcript containing multiple sentences
                if self.lowercase:
                    tokens = [w.lower() for w in tokens]

                # split into sentences
                words_list = [[]]  # a list of sentences
                for w in tokens:
                    words_list[-1].append(w)
                    if w in punctuation:
                        words_list.append([])

                # collect words + lm_tags
                for words in words_list:
                    if len(words) <= num_masked_per_sentence:
                        num_skipped += 1
                        continue

                    masked_words = random.sample(words, k=num_masked_per_sentence)
                    lm_tags = [w if w in masked_words else 'O' for w in words]  # TODO maybe don't use 'O' ?

                    if len(words) > self.params.max_sentence_length:
                        continue

                    sentences.append((words, lm_tags))

        print(f'WARNING: Skipped {num_skipped} sentences which are shorter than number of words to mask.')

        return sentences

    # --------------------------------------------------------- interface with Allen NLP toolkit

    @staticmethod
    def make_lm_mask(sentence):
        """
        return a one-hot list where hot value marks masked word to be predicted
        :param sentence: a tuple with structure (words, labels)
        :return: list of ones and zeros where ones mark word to be masked and predicted
        """
        lm_tags = sentence[1]
        res = [0 if lm_tag == 'O' else 1 for lm_tag in lm_tags]  # TODO test
        return res

    def _text_to_instance(self,
                          tokens: List[Token],
                          lm_mask: List[int],  # masked language modeling
                          lm_tags: List[str] = None) -> Instance:

        # to word-pieces
        word_pieces, offsets, start_offsets = wordpiece_tokenize_input([t.text for t in tokens],
                                                                       self.bert_tokenizer,
                                                                       self.lowercase)
        new_mask = convert_lm_mask_to_wordpiece_lm_mask(lm_mask, offsets)

        # In order to override the indexing mechanism, we need to set the `text_id`
        # attribute directly. This causes the indexing to use this id.
        # new_tokens = [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in word_pieces]
        new_tokens = [Token(t) for t in word_pieces]

        # WARNING:
        # setting text_id causes tokens not to be found by Allen Vocabulary.
        # allen nlp bert test case doesn't use token fields, but instead uses:
        # tokens = fields["metadata"]["words"]

        text_field = TextField(new_tokens, self.token_indexers)
        mask_indicator = SequenceLabelField(new_mask, text_field)
        fields = {'tokens': text_field,
                  'mask_indicator': mask_indicator}

        # metadata
        metadata_dict: Dict[str, Any] = {}

        if all([x == 0 for x in lm_mask]):
            raise ValueError('Mask indicator contains zeros only. ')
        else:
            masked_words = [t.text for m, t in zip(lm_mask, tokens) if m]

        metadata_dict["offsets"] = start_offsets
        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["masked_words"] = masked_words
        metadata_dict["masked_indices"] = lm_mask  # mask is list containing zeros and ones

        if lm_tags:
            new_lm_tags = convert_lm_tags_to_wordpiece_lm_tags(lm_tags, offsets)
            fields['lm_tags'] = SequenceLabelField(new_lm_tags, text_field)
            metadata_dict["gold_lm_tags"] = lm_tags  # non word-piece tags

        fields["metadata"] = MetadataField(metadata_dict)

        return Instance(fields)

    def make_instances(self, sentences) -> Iterator[Instance]:
        """
        because lazy is by default False, return a list rather than a generator.
        When lazy=False, the generator would be converted to a list anyway.

        roughly equivalent to Allen NLP toolkit dataset.read()

        """
        res = []
        for sentence in sentences:
            words = sentence[0]
            lm_mask = self.make_lm_mask(sentence)  # 1s and 0s depending on whether word is to be predicted
            lm_tags = sentence[1]
            # to instance
            instance = self._text_to_instance([Token(word) for word in words],
                                              lm_mask,
                                              lm_tags)
            res.append(instance)
        return res