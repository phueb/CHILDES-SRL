from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    data_tools = root / 'data_tools'
    perl = root / 'perl'


class Data:
    min_seq_length = 3
    max_seq_length = 128
    childes_symbols = {'[NAME]', '[PLACE]', '[MISC]'}


class Example:
    eval_interval = 10_000  # number of steps after which to evaluate performance
    feedback_interval = 1000  # number of steps after which to print feedback to console

