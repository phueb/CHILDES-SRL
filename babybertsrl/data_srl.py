import numpy as np
from typing import Iterator, List
from pathlib import Path

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl.word_pieces import wordpiece
from babybertsrl.word_pieces import convert_verb_indices_to_wordpiece_indices
from babybertsrl.word_pieces import convert_tags_to_wordpiece_tags


class DataSRL:

    def __init__(self,
                 params,
                 data_path: Path,
                 bert_tokenizer: BertTokenizer,
                 ):
        """
        loads propositions from file and puts them in Allen NLP toolkit instances format
        for training a BERT-based SRL tagger.
        designed to use with conll-05 style formatted SRL data.
        """

        self.params = params

        # load propositions
        self.propositions = self.get_propositions_from_file(data_path)
        lengths = [len(s[0]) for s in self.propositions]
        print('Found {:,} utterances'.format(len(self.propositions)))
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')
        print()

        self.bert_tokenizer = bert_tokenizer

        # instances
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.instances = self.make_instances(self.propositions)

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
    def make_verb_indices(proposition):
        """
        return a one-hot list where hot value marks verb
        :param proposition: a tuple with structure (words, predicate, labels)
        :return: one-hot list, [sentence length]
        """
        num_w_in_proposition = len(proposition[0])
        res = [int(i == proposition[1]) for i in range(num_w_in_proposition)]

        if all([x == 0 for x in res]):
            raise ValueError('Verb indicator contains zeros only. ')

        return res

    def _text_to_instance(self,
                          srl_in: List[str],
                          srl_verb_indices: List[int],
                          srl_tags: List[str],
                          ) -> Instance:

        # to word-pieces
        srl_in_word_pieces, offsets, start_offsets = wordpiece(srl_in,
                                                               self.bert_tokenizer,
                                                               lowercase_input=False)
        srl_tags_word_pieces = convert_tags_to_wordpiece_tags(srl_tags, offsets)
        verb_indices_word_pieces = convert_verb_indices_to_wordpiece_indices(srl_verb_indices, offsets)

        # compute verb
        verb_index = srl_verb_indices.index(1)
        verb = srl_in_word_pieces[verb_index]

        # metadata only has whole words
        metadata_dict = dict()
        metadata_dict['offsets'] = start_offsets
        metadata_dict['srl_in'] = srl_in   # previously called "words"
        metadata_dict['verb'] = verb
        metadata_dict['verb_index'] = verb_index  # must be an integer
        metadata_dict['gold_srl_tags'] = srl_tags  # non word-piece tags

        # fields
        tokens = [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in srl_in_word_pieces]
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
            srl_in = proposition[0]
            srl_verb_indices = self.make_verb_indices(proposition)
            srl_tags = proposition[2]

            # to instance
            instance = self._text_to_instance(srl_in,
                                              srl_verb_indices,
                                              srl_tags)
            yield instance