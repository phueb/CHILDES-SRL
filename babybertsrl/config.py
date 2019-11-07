from pathlib import Path


class LocalDirs:
    root = Path(__file__).parent.parent
    data = root / 'data'


class Global:
    debug = False


class Data:
    bert_name = 'bert-base-uncased'  # if 'uncased', input is lowercased


class Eval:
    loss_interval = 100



