from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    data_tools = root / 'data_tools'


class Data:
    min_input_length = 3
    max_input_length = 128
    train_prob = 0.8  # probability that utterance is train utterance


class Eval:
    interval = 10_000
    test_sentences = False
    train_split = False
    print_perl_script_output = False  # happens at every batch so not very useful

    probing_names = [
        'dummy',
        'agreement_across_adjectives',
        'agreement_across_PP',
        'agreement_across_RC'
    ]



