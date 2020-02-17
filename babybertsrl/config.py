from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    data_tools = root / 'data_tools'


class Global:
    debug = False


class Data:
    min_utterance_length = 3  # used during language modeling task
    max_utterance_length = 128  # used during language modeling task
    train_prob = 0.8  # probability that utterance is train utterance


class Eval:
    loss_interval = 100



