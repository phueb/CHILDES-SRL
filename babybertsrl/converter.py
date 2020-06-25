from typing import Iterator, List, Tuple, Union
import numpy as np

from pytorch_pretrained_bert.tokenization import WordpieceTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl.word_pieces import wordpiece, convert_tags_to_wordpiece_tags, convert_verb_indices_to_wordpiece_indices


def mask_one_element(elements: List[str],
                     masked_id: int,
                     ) -> Tuple[List[str], List[int]]:

    masked = ['[MASK]' if i == masked_id else elements[i] for i in range(len(elements))]
    mask = [1 if i == masked_id else 0 for i in range(len(elements))]

    if all([x == 0 for x in mask]):
        raise ValueError('Mask indicator contains zeros only. ')

    return masked, mask


class ConverterMLM:

    def __init__(self,
                 params,
                 wordpiece_tokenizer: WordpieceTokenizer,
                 ):
        """
        converts utterances into Allen NLP toolkit instances format
        for training with BERT.
        designed to use with CHILDES sentences

        """

        self.params = params
        self.wordpiece_tokenizer = wordpiece_tokenizer
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}  # specifies how a token is indexed

    def _text_to_instance(self,
                          mlm_in: List[str],
                          mlm_in_wp: List[str],
                          mlm_tags: List[str],
                          mlm_tags_wp: List[str],
                          start_offsets: List[int],
                          mlm_mask_wp: List[int],
                          ) -> Instance:

        # meta data only has whole words
        metadata_dict = dict()
        metadata_dict['start_offsets'] = start_offsets
        metadata_dict['in'] = mlm_in
        metadata_dict['gold_tags'] = mlm_tags  # is just a copy of the input without the mask

        # fields
        tokens = [Token(t, text_id=self.wordpiece_tokenizer.vocab[t]) for t in mlm_in_wp]
        text_field = TextField(tokens, self.token_indexers)

        assert len(mlm_in_wp) == len(mlm_tags_wp)

        fields = {'tokens': text_field,
                  'indicator': SequenceLabelField(mlm_mask_wp, text_field),
                  'tags': SequenceLabelField(mlm_tags_wp, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self,
                       utterances:  List[List[str]],
                       ) -> List[Instance]:
        """
        convert on utterance into possibly multiple Allen NLP instances



        # TODO 1. convert wo wp 2. mask word in whole-word sequence 3. use offsets to compute wp sequence with [MASK]

        """

        res = []
        for mlm_in in utterances:

            # to word-pieces (do this BEFORE masking)  TODO test
            mlm_in_wp, offsets, start_offsets = wordpiece(mlm_in,
                                                          self.wordpiece_tokenizer,
                                                          lowercase_input=False)
            mlm_tags = mlm_in.copy()
            mlm_tags_wp = mlm_in_wp.copy()

            # collect each multiple times, each time with a different masked word
            num_wps = len(mlm_in_wp)
            num_masked = min(num_wps, self.params.num_masked)
            for masked_id in np.random.choice(num_wps, num_masked, replace=False):
                # mask
                mlm_in_wp, mlm_mask_wp = mask_one_element(mlm_in_wp, masked_id)
                # to instance
                instance = self._text_to_instance(mlm_in,
                                                  mlm_in_wp,
                                                  mlm_tags,
                                                  mlm_tags_wp,
                                                  start_offsets,
                                                  mlm_mask_wp)
                res.append(instance)

        print(f'With num_masked={self.params.num_masked}, made {len(res):>9,} MLM instances')

        return res

    def make_probing_instances(self,
                               utterances:  List[List[str]],
                               ) -> List[Instance]:
        """
        convert on utterance into exactly one Allen NLP instances - WITHOUT MASKING (assuming masking is already done)

        """

        res = []
        for mlm_in in utterances:
            # to word-pieces (do this BEFORE masking)  TODO test
            mlm_in_wp, offsets, start_offsets = wordpiece(mlm_in,
                                                          self.wordpiece_tokenizer,
                                                          lowercase_input=False)

            mlm_tags = mlm_in  # irrelevant for probing
            mlm_tags_wp = mlm_in_wp  # irrelevant for probing

            # mask
            masked_id = mlm_in.index('[MASK]')
            mlm_in_wp, mlm_mask_wp = mask_one_element(mlm_in_wp, masked_id)
            #  [MASK] symbol is not in output vocab - convert
            mlm_tags_wp = [w if w != '[MASK]' else '[UNK]' for w in mlm_tags_wp]

            # to instance
            instance = self._text_to_instance(mlm_in,
                                              mlm_in_wp,
                                              mlm_tags,
                                              mlm_tags_wp,
                                              start_offsets,
                                              mlm_mask_wp)
            res.append(instance)

        print(f'Without masking, made {len(res):>9,} probing MLM instances')

        return res


class ConverterSRL:

    def __init__(self,
                 params,
                 wordpiece_tokenizer: WordpieceTokenizer,
                 ):
        """
        converts propositions into Allen NLP toolkit instances format
        for training a BERT-based SRL tagger.
        designed to use with conll-05 style formatted SRL data.
        """

        self.params = params
        self.wordpiece_tokenizer = wordpiece_tokenizer
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

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
        srl_in_wp, offsets, start_offsets = wordpiece(srl_in,
                                                      self.wordpiece_tokenizer,
                                                      lowercase_input=False)
        srl_tags_wp = convert_tags_to_wordpiece_tags(srl_tags, offsets)
        verb_indices_wp = convert_verb_indices_to_wordpiece_indices(srl_verb_indices, offsets)

        # compute verb
        verb_index = srl_verb_indices.index(1)
        verb = srl_in_wp[verb_index]

        # metadata only has whole words
        metadata_dict = dict()
        metadata_dict['start_offsets'] = start_offsets
        metadata_dict['in'] = srl_in   # previously called "words"
        metadata_dict['verb'] = verb
        metadata_dict['verb_index'] = verb_index  # must be an integer
        metadata_dict['gold_tags'] = srl_tags  # non word-piece tags

        # fields
        tokens = [Token(t, text_id=self.wordpiece_tokenizer.vocab[t]) for t in srl_in_wp]
        text_field = TextField(tokens, self.token_indexers)

        fields = {'tokens': text_field,
                  'indicator': SequenceLabelField(verb_indices_wp, text_field),
                  'tags': SequenceLabelField(srl_tags_wp, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, propositions: List[Tuple[List[str], int, List[str]]],
                       ) -> List[Instance]:
        """
        roughly equivalent to Allen NLP toolkit dataset.read().
        return a list rather than a generator,
         because DataIterator requires being able to iterate multiple times to implement multiple epochs.

        """
        res = []
        for proposition in propositions:
            srl_in = proposition[0]
            srl_verb_indices = self.make_verb_indices(proposition)
            srl_tags = proposition[2]

            # to instance
            instance = self._text_to_instance(srl_in,
                                              srl_verb_indices,
                                              srl_tags)
            res.append(instance)

        print(f'Made {len(res):>9,} SRL instances')

        return res