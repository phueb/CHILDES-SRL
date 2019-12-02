from pathlib import Path


class LocalDirs:
    root = Path(__file__).parent.parent
    data = root / 'data'


class Global:
    debug = False


class Data:
    min_utterance_length = 3  # used during language modeling task
    replace = {''}


class Eval:
    loss_interval = 100



