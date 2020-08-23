from typing import List, Tuple

from training.word_pieces import(
    convert_words_to_wordpieces,
    convert_bio_tags_to_wordpieces,
    convert_verb_indices_to_wordpiece_indices
)


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
    srl_in_wp, offsets, start_offsets = convert_words_to_wordpieces(srl_in, self.wordpiece_tokenizer)
    srl_tags_wp = convert_bio_tags_to_wordpieces(srl_tags, offsets)
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
    metadata_dict['gold_tags_wp'] = srl_tags_wp

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