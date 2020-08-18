import numpy as np
from typing import List
from pathlib import Path
import random
from collections import OrderedDict

from babybertsrl import configs


# when lower-casing, do not lower-case upper-cased symbols
upper_cased = configs.Data.special_symbols + configs.Data.childes_symbols  # order matters


def load_words_from_vocab_file(vocab_file: Path,
                               col: int = 0):

    res = []
    with vocab_file.open("r", encoding="utf-8") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            token = line.split()[col]

            # exclude word with non-ASCII characters
            if [True for c in token if ord(c) > 127]:
                continue

            res.append(token)
    return res


def load_vocab(childes_vocab_file: Path,
               google_vocab_file: Path,
               vocab_size: int,  # childes-vocab, not total vocab
               google_vocab_rule: str) -> OrderedDict:

    childes_vocab = load_words_from_vocab_file(childes_vocab_file, col=1)[:vocab_size]
    google_vocab = load_words_from_vocab_file(google_vocab_file, col=0)

    # exclude any google wordpieces not in CHILDES vocab, but leave non-start wordpieces (e.g. ##s)
    google_vocab_cleaned = [w for w in google_vocab
                            if w in set(childes_vocab) or w.startswith('##')]

    # init
    to_index = configs.Data.special_symbols + configs.Data.childes_symbols

    # add from childes vocab
    if google_vocab_rule == 'inclusive':
        to_index += set(childes_vocab + google_vocab_cleaned)
    elif google_vocab_rule == 'exclusive':
        to_index += google_vocab_cleaned
    elif google_vocab_rule == 'excluded':
        to_index += childes_vocab
    else:
        raise AttributeError('Invalid arg to "google_vocab_rule".')

    # index
    res = OrderedDict()
    index = 0
    for token in to_index:
        if token in res:
            # happens for symbols
            continue
        res[token] = index
        index += 1

    assert len(set(res)) == len(res)
    assert res['[MASK]'] == configs.Data.mask_vocab_id
    assert index == len(res), (index, len(res))

    return res


def split(data: List, seed: int = 2):

    random.seed(seed)

    train = []
    devel = []
    test = []

    for i in data:

        if random.choices([True, False],
                          weights=[configs.Data.train_prob, 1 - configs.Data.train_prob])[0]:
            train.append(i)
        else:
            if random.choices([True, False], weights=[0.5, 0.5])[0]:
                devel.append(i)
            else:
                test.append(i)

    print(f'num train={len(train):,}')
    print(f'num devel={len(devel):,}')
    print(f'num test ={len(test):,}')

    return train, devel, test


def load_utterances_from_file(file_path: Path,
                              verbose: bool = False,
                              allow_discard: bool = False) -> List[List[str]]:
    """
    load utterances for language modeling from text file
    """

    print(f'Loading {file_path}')

    res = []
    punctuation = {'.', '?', '!'}
    num_too_small = 0
    num_too_large = 0
    with file_path.open('r') as f:

        for line in f.readlines():

            # tokenize transcript
            transcript = line.strip().split()  # a transcript containing multiple utterances
            transcript = [w for w in transcript]

            # split transcript into utterances
            utterances = [[]]
            for w in transcript:
                utterances[-1].append(w)
                if w in punctuation:
                    utterances.append([])

            # collect utterances
            for utterance in utterances:

                if not utterance:  # during probing, parsing logic above may produce empty utterances
                    continue

                # check  length
                if len(utterance) < configs.Data.min_seq_length and allow_discard:
                    num_too_small += 1
                    continue
                if len(utterance) > configs.Data.max_seq_length and allow_discard:
                    num_too_large += 1
                    continue

                # lower-case
                if configs.Data.uncased:
                    utterance = [w if w in upper_cased else w.lower()
                                 for w in utterance]

                res.append(utterance)

    print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {configs.Data.min_seq_length}.')
    print(f'WARNING: Skipped {num_too_large} utterances which are larger than {configs.Data.max_seq_length}.')

    if verbose:
        lengths = [len(u) for u in res]
        print('Found {:,} utterances'.format(len(res)))
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')

    return res


def load_propositions_from_file(file_path: Path,
                                verbose: bool = False):
    """
    Read tokenized propositions from file.
    File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
    Return:
        A list with elements of structure [[words], predicate position, [labels]]
    """

    print(f'Loading {file_path}')

    num_too_small = 0
    num_too_large = 0
    res = []
    with file_path.open('r') as f:

        for line in f.readlines():

            inputs = line.strip().split('|||')
            left_input = inputs[0].strip().split()
            right_input = inputs[1].strip().split()

            # predicate
            predicate_index = int(left_input[0])

            # words + labels
            words = left_input[1:]
            labels = right_input

            # check  length
            if len(words) <= configs.Data.min_seq_length:
                num_too_small += 1
                continue
            if len(words) > configs.Data.max_seq_length:
                num_too_large += 1
                continue

            # lower-case
            if configs.Data.uncased:
                words = [w if w in upper_cased else w.lower()
                         for w in words]

            res.append((words, predicate_index, labels))

    print(f'WARNING: Skipped {num_too_small} propositions which are shorter than {configs.Data.min_seq_length}.')
    print(f'WARNING: Skipped {num_too_large} propositions which are larger than {configs.Data.max_seq_length}.')

    if verbose:
        lengths = [len(p[0]) for p in res]
        print('Found {:,} propositions'.format(len(res)))
        print(f'Max    proposition length: {np.max(lengths):.2f}')
        print(f'Mean   proposition length: {np.mean(lengths):.2f}')
        print(f'Median proposition length: {np.median(lengths):.2f}')

    return res
