from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple
from pytorch_pretrained_bert.tokenization import WordpieceTokenizer

probing_names = [
        'dummy',
        'agreement_across_adjectives',
        'agreement_across_PP',
        'agreement_across_RC',
        'agreement_in_question'
    ]

special_symbols = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']  # order matters
childes_symbols = ['[NAME]', '[PLACE]', '[MISC]']  # do not word-piece or lower-case these
all_symbols = special_symbols + childes_symbols

probing_path = '../data/probing'
google_vocab_file = "../data/bert-base-uncased-vocab.txt"
childes_vocab_file = '../data/childes-20191206_vocab.txt'
# https://github.com/phueb/BabyBertSRL/blob/master/data/childes-20191206_vocab.txt


def load_utterances_from_file(file_path: Path,
                              ) -> List[List[str]]:
    """
    load utterances for language modeling from text file
    """

    min_input_length = 3
    max_input_length = 128

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

                # check  length
                if len(utterance) < min_input_length:
                    num_too_small += 1
                    continue
                if len(utterance) > max_input_length:
                    num_too_large += 1
                    continue

                res.append(utterance)

    print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {min_input_length}.')
    print(f'WARNING: Skipped {num_too_large} utterances which are larger than {max_input_length}.')

    return res


def convert_words_to_wordpieces(tokens: List[str],
                                wordpiece_tokenizer,
                                lowercase_input: bool,
                                ) -> Tuple[List[str], List[int], List[int]]:
    """
    Convert a list of tokens to wordpiece tokens and offsets, as well as adding
    BERT CLS and SEP tokens to the begining and end of the sentence.

    Returns
    -------
    wordpieces : List[str]
        The BERT wordpieces from the words in the sentence.
    end_offsets : List[int]
        Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
        results in the end wordpiece of each word being chosen.
    start_offsets : List[int]
        Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
        results in the start wordpiece of each word being chosen.
    """
    word_piece_tokens: List[str] = []
    end_offsets = []
    start_offsets = []
    cumulative = 0
    for token in tokens:
        if lowercase_input and token not in all_symbols:
            token = token.lower()
        word_pieces = wordpiece_tokenizer.tokenize(token)
        start_offsets.append(cumulative + 1)
        cumulative += len(word_pieces)
        end_offsets.append(cumulative)
        word_piece_tokens.extend(word_pieces)

    wordpieces = ['[CLS]'] + word_piece_tokens + ['[SEP]']

    return wordpieces, end_offsets, start_offsets


def load_words_from_vocab_file(vocab_file: Path,
                               col: int =0):

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


def to_bert_format(utterances:  List[List[str]],
                           ) -> List[List[str]]:
    """
    convert each utterance into exactly one word-pieced sequence.
    masking is assumed to be already done.

    """

    wordpiece_tokenizer = make_wordpiece_tokenizer()

    res = []
    for utterance in utterances:
        # to word-pieces (do this BEFORE masking)
        mlm_in_wp, offsets, start_offsets = convert_words_to_wordpieces(utterance,
                                                                        wordpiece_tokenizer,
                                                                        lowercase_input=True)
        res.append(mlm_in_wp)

    print(f'Made {len(res):>9,} probing sequences')

    return res


def make_wordpiece_tokenizer():
    """
    uses google vocab as starting point,
    but also adds any word from 4k most frequent words in CHILDES that is not in google vocab
    """
    childes_vocab = load_words_from_vocab_file(Path(childes_vocab_file), col=1)[:4000]
    google_vocab = load_words_from_vocab_file(Path(google_vocab_file), col=0)
    # make vocab index for tokenizer
    to_index = special_symbols + childes_symbols + list(set(childes_vocab + google_vocab))
    vocab = OrderedDict()
    index = 0
    for token in to_index:
        if token in vocab:
            # happens for symbols
            continue
        vocab[token] = index
        index += 1
    assert vocab['[PAD]'] == 0
    assert vocab['[UNK]'] == 1
    assert vocab['[CLS]'] == 2
    assert vocab['[SEP]'] == 3
    assert vocab['[MASK]'] == 4

    return WordpieceTokenizer(vocab)


def do_probing(step):
    """
    predict masked word given test sentences belong ing to various probing tasks
    """
    for name in probing_names:
        # load probing sequences
        probing_data_path = Path(probing_path) / f'{name}.txt'
        if not probing_data_path.exists():
            print(f'WARNING: {probing_data_path} does not exist', flush=True)
            continue
        print(f'Starting probing with task={name}', flush=True)
        probing_utterances = load_utterances_from_file(probing_data_path)
        input_sequences = to_bert_format(probing_utterances)

        # sanity check
        for s in input_sequences:
            print(s)

        # get predictions from BERT - in words, not integers, for saving to file
        # TODO: make sure to predict a word only at [MASK], all other words should remain identical to input
        predicted_sequences = get_predictions(input_sequences)

        # save predictions to file
        out_path = Path('predictions') / f'probing_{name}_results_{step}.txt'
        out_path.parent.mkdir()
        print(f'Saving MLM prediction results to {out_path}')
        with out_path.open('w') as f:
            for a, b in zip(input_sequences, predicted_sequences):
                assert len(a) == len(b)
                for ai, bi in zip(a, b):  # careful, zips over shortest list
                    line = f'{ai:>20} {bi:>20}'
                    f.write(line + '\n')
                f.write('\n')


if __name__ == '__main__':
    do_probing(step=0)
