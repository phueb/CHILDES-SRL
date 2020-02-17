from typing import Iterator, List, Tuple, Optional
import numpy as np

from pytorch_pretrained_bert.tokenization import WordpieceTokenizer

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField

from babybertsrl.word_pieces import wordpiece, convert_tags_to_wordpiece_tags, convert_verb_indices_to_wordpiece_indices
from babybertsrl.word_pieces import convert_mlm_mask_to_wordpiece_mlm_mask


def prepare_utterance_for_instance(words: List[str],
                                   masked_id: int,
                                   ) -> Tuple[List[str], List[int], List[str]]:

    mlm_in = ['[MASK]' if i == masked_id else words[i] for i in range(len(words))]
    mlm_mask = [1 if i == masked_id else 0 for i in range(len(words))]
    mlm_tags = words

    if all([x == 0 for x in mlm_mask]):
        raise ValueError('Mask indicator contains zeros only. ')

    return mlm_in, mlm_mask, mlm_tags


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
                          mlm_mask: List[int],
                          mlm_tags: List[str],
                          ) -> Instance:

        # to word-pieces
        mlm_in_word_pieces, offsets, start_offsets = wordpiece(mlm_in,
                                                               self.wordpiece_tokenizer,
                                                               lowercase_input=False)
        mlm_tags_word_pieces, _, _ = wordpiece(mlm_tags,
                                               self.wordpiece_tokenizer,
                                               lowercase_input=False)
        mlm_mask_word_pieces = convert_mlm_mask_to_wordpiece_mlm_mask(mlm_mask, offsets)

        # meta data only has whole words
        metadata_dict = dict()
        metadata_dict['start_offsets'] = start_offsets
        metadata_dict['in'] = mlm_in
        metadata_dict['masked_indices'] = mlm_mask  # mask is list containing zeros and ones
        metadata_dict['gold_tags'] = mlm_tags  # is just a copy of the input without the mask

        # fields
        tokens = [Token(t, text_id=self.wordpiece_tokenizer.vocab[t]) for t in mlm_in_word_pieces]
        text_field = TextField(tokens, self.token_indexers)

        if len(mlm_in_word_pieces) != len(mlm_tags_word_pieces):
            # the code does not yet support custom word-pieces in vocabulary,
            # because it does not handle case when masked word is split into word-pieces.
            # In such a case, input and output length are mismatched.
            # The output is longer, because it contains more than 1 piece for the masked whole word in the input.
            raise UserWarning('A word-piece split word was masked. Word pieces are not supported')

        fields = {'tokens': text_field,
                  'indicator': SequenceLabelField(mlm_mask_word_pieces, text_field),
                  'tags': SequenceLabelField(mlm_tags_word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, utterances:  List[List[str]],
                       ) -> Iterator[Instance]:
        """
        convert on utterance into possibly multiple Allen NLP instances

        """
        for utterance in utterances:

            # collect each multiple times, each time with a different masked word
            utterance_length = len(utterance)
            num_masked = min(utterance_length, self.params.num_masked)
            for masked_id in np.random.choice(utterance_length, num_masked, replace=False):

                # collect instance
                mlm_in, mlm_mask, mlm_tags = prepare_utterance_for_instance(utterance, masked_id)
                instance = self._text_to_instance(mlm_in, mlm_mask, mlm_tags)

                yield instance

    def num_instances(self, utterances:  List[List[str]],
                      ) -> int:
        """
        must mirror logic in make_instances() to provide accurate result.
        """
        res = 0
        for utterance in utterances:

            utterance_length = len(utterance)
            num_masked = min(utterance_length, self.params.num_masked)
            res += num_masked
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
        srl_in_word_pieces, offsets, start_offsets = wordpiece(srl_in,
                                                               self.wordpiece_tokenizer,
                                                               lowercase_input=False)
        srl_tags_word_pieces = convert_tags_to_wordpiece_tags(srl_tags, offsets)
        verb_indices_word_pieces = convert_verb_indices_to_wordpiece_indices(srl_verb_indices, offsets)

        # compute verb
        verb_index = srl_verb_indices.index(1)
        verb = srl_in_word_pieces[verb_index]

        # metadata only has whole words
        metadata_dict = dict()
        metadata_dict['start_offsets'] = start_offsets
        metadata_dict['in'] = srl_in   # previously called "words"
        metadata_dict['verb'] = verb
        metadata_dict['verb_index'] = verb_index  # must be an integer
        metadata_dict['gold_tags'] = srl_tags  # non word-piece tags

        # fields
        tokens = [Token(t, text_id=self.wordpiece_tokenizer.vocab[t]) for t in srl_in_word_pieces]
        text_field = TextField(tokens, self.token_indexers)

        fields = {'tokens': text_field,
                  'indicator': SequenceLabelField(verb_indices_word_pieces, text_field),
                  'tags': SequenceLabelField(srl_tags_word_pieces, text_field),
                  'metadata': MetadataField(metadata_dict)}

        return Instance(fields)

    def make_instances(self, propositions: List[Tuple[List[str], int, List[str]]],
                       ) -> Iterator[Instance]:
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