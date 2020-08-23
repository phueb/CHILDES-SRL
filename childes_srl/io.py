import numpy as np
from typing import List, Set, Tuple, Optional
from pathlib import Path

from childes_srl import configs


def load_mlm_data(file_path: Path,
                  verbose: bool = False,
                  uncased: bool = False,
                  special_tokens: Optional[Set[str]] = None,
                  allow_discard: bool = False) -> List[List[str]]:
    """
    load CHILDES utterances for adding SR-labels
    """

    if special_tokens is None:
        special_tokens = configs.Data.childes_symbols

    assert file_path.exists()
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
                if uncased:
                    utterance = [w if w in special_tokens else w.lower()
                                 for w in utterance]

                res.append(utterance)

    if num_too_small or num_too_large:
        print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {configs.Data.min_seq_length}.')
        print(f'WARNING: Skipped {num_too_large} utterances which are larger than {configs.Data.max_seq_length}.')

    if verbose:
        lengths = [len(u) for u in res]
        print('Found {:,} utterances'.format(len(res)))
        print(f'Min    utterance length: {np.min(lengths):.2f}')
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')

    return res


def load_srl_data(file_path: Path,
                  verbose: bool = False,
                  uncased: bool = False,
                  special_tokens: Optional[Set[str]] = None,
                  ) -> List[Tuple]:
    """
    Read tokenized propositions from file.
    File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
    Return:
        A list with elements of structure [[words], predicate position, [labels]]
    """

    if special_tokens is None:
        special_tokens = configs.Data.childes_symbols

    assert file_path.exists()
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
            if uncased:
                words = [w if w in special_tokens else w.lower()
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
