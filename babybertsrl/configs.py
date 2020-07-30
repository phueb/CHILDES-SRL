from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    data_tools = root / 'data_tools'


class Data:
    min_input_length = 3
    max_input_length = 128
    train_prob = 0.8  # probability that utterance is assigned to train split
    special_symbols = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']  # order matters
    childes_symbols = ['[NAME]', '[PLACE]', '[MISC]']

    mask_vocab_id = 4


class Training:
    feedback_interval = 100
    ignored_index = -1 # any ids in argument "tags" to cross-entropy fn are ignored
    debug = False


class Eval:
    interval = 10_000
    test_sentences = False
    train_split = False
    print_perl_script_output = False  # happens at every batch so not very useful
    probe_at_step_zero = True
    probe_at_end = False

    probing_names = [
        'dummy',
        'agreement_across_adjectives',
        'agreement_across_PP',
        'agreement_across_RC',
        'agreement_in_1_verb_question',
        'agreement_in_2_verb_question',
    ]


class Wordpieces:
    verbose = False
    warn_on_mismatch = False

