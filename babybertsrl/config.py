from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    data_tools = root / 'data_tools'


class Global:
    debug = False


class Data:
    min_input_length = 3
    max_input_length = 128
    train_prob = 0.8  # probability that utterance is train utterance


class Eval:
    interval = 1000
    test_sentences = False
    train_split = True



