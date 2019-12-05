import numpy as np
from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl import config
from babybertsrl.word_pieces import wordpiece_tokenize
from babybertsrl.word_pieces import convert_verb_indices_to_wordpiece_indices
from babybertsrl.word_pieces import convert_tags_to_wordpiece_tags


class Data:

    def __init__(self,
                 params,
                 train_data_path: Path,
                 dev_data_path: Path,
                 vocab_path_name: Optional[str] = None,
                 ):
        """
        loads propositions from file and puts them in Allen NLP toolkit instances format
        for training with BERT.
        designed to use with conll-05 data
        """

        self.params = params
        if vocab_path_name:
            self.bert_tokenizer = BertTokenizer(vocab_path_name, do_basic_tokenize=False)
            self.lowercase = self.bert_tokenizer.basic_tokenizer.do_lower_case
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(config.Data.bert_name)
            self.lowercase = 'uncased' in config.Data.bert_name

        # load propositions
        self.train_propositions = self.get_propositions_from_file(train_data_path)
        self.dev_propositions = self.get_propositions_from_file(dev_data_path)

        # print info
        print('Found {:,} training propositions ...'.format(self.num_train_propositions))
        print('Found {:,} dev propositions ...'.format(self.num_dev_propositions))
        print()
        for name, propositions in zip(['train', 'dev'],
                                      [self.train_propositions, self.dev_propositions]):
            lengths = [len(p[0]) for p in propositions]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
            print()

        # training with Allen NLP toolkit
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.train_instances = self.make_instances(self.train_propositions)
        self.dev_instances = self.make_instances(self.dev_propositions)

    @property
    def num_train_propositions(self):
        return len(self.train_propositions)

    @property
    def num_dev_propositions(self):
        return len(self.dev_propositions)

    def get_propositions_from_file(self, file_path):
        """
        Read tokenized propositions from file.
          File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
          Return:
            A list with elements of structure [[words], predicate, [labels]]
        """
        propositions = []
        with file_path.open('r') as f:

            for line in f.readlines():

                inputs = line.strip().split('|||')
                left_input = inputs[0].strip().split()
                right_input = inputs[1].strip().split()

                if self.lowercase:
                    left_input = [w.lower() for w in left_input]

                # predicate
                predicate_pos = int(left_input[0])

                # words + labels
                words = left_input[1:]
                labels = right_input

                if len(words) > self.params.max_sentence_length:
                    continue

                propositions.append((words, predicate_pos, labels))

        return propositions

    # --------------------------------------------------------- interface with Allen NLP toolkit

    @staticmethod
    def make_predicate_one_hot(proposition):
        """
        return a one-hot list where hot value marks verb
        :param proposition: a tuple with structure (words, predicate, labels)
        :return: one-hot list, [sentence length]
        """
        num_w_in_proposition = len(proposition[0])
        res = [int(i == proposition[1]) for i in range(num_w_in_proposition)]
        return res

    def _text_to_instance(self,
                          srl_in: List[str],
                          srl_verb_indices: List[int],
                          srl_tags: List[str],
                          ) -> Instance:

        # to word-pieces
        srl_in_word_pieces, offsets, start_offsets = wordpiece_tokenize(srl_in,
                                                                        self.bert_tokenizer,
                                                                        self.lowercase)
        srl_tags_word_pieces = convert_tags_to_wordpiece_tags(srl_tags, offsets)
        verb_indices_word_pieces = convert_verb_indices_to_wordpiece_indices(srl_verb_indices, offsets)

        # compute verb
        if all([x == 0 for x in srl_verb_indices]):
            raise ValueError('Verb indicator contains zeros only. ')
        else:
            verb_index = srl_verb_indices.index(1)
            verb = srl_in_word_pieces[verb_index]

        # metadata only has whole words
        metadata_dict = dict()
        metadata_dict['offsets'] = start_offsets
        metadata_dict['srl_in'] = srl_in   # TODO previously called "words"
        metadata_dict['verb'] = verb
        metadata_dict['verb_indices'] = srl_verb_indices  # TODO previously called "verb index"
        metadata_dict['gold_srl_tags'] = srl_tags  # non word-piece tags

        # fields
        tokens = [Token(t) for t in srl_in_word_pieces]
        text_field = TextField(tokens, self.token_indexers)

        fields = {'tokens': text_field,
                  'verb_indicator': SequenceLabelField(verb_indices_word_pieces, text_field),
                  'srl_tags': SequenceLabelField(srl_tags_word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, propositions) -> Iterator[Instance]:
        """
        roughly equivalent to Allen NLP toolkit dataset.read()

        """
        for proposition in propositions:
            words = proposition[0]
            predicate_one_hot = self.make_predicate_one_hot(proposition)
            tags = proposition[2]

            # to instance
            instance = self._text_to_instance([Token(word) for word in words],
                                              predicate_one_hot,
                                              tags)
            yield instance