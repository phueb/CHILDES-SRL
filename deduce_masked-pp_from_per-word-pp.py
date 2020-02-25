

# for evaluating test sentence prediction performance, it would be useful to know pp of masked word only,
# rather than average pp across all words (as is computed by BERT)


def deduce_masked_pp(per_word_pp, utterance_length):
    """assume only 1 word is masked in utterance"""
    num_non_masked = utterance_length - 1
    return per_word_pp * utterance_length - num_non_masked


length = 4
per_word_pp = 3.4
masked_pp = deduce_masked_pp(per_word_pp, length)
assert (masked_pp + length - 1) / length == per_word_pp