from pathlib import Path
import socket


class RemoteDirs:
    root = Path('/media/research_data') / 'BabyBERTSRL'
    runs = root / 'runs'
    data = root / 'data'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'babybertsrl'
    runs = root / '{}_runs'.format(src.name)
    data = root / 'data'


class Global:
    debug = False


class Data:
    """
    # the letter "O" is used to mark any words without labels as "outside" in conll05
    """

    lowercase = True  # True gives strong performance boost
    bio_tags = True  # the "O" tag is still used to label "outside" words if set to False

    unk_word = '<UNKNOWN>'  # TODO use this for test data - but don't use an UNKNOWN_LABEL

    pad_word = '<PAD>'
    pad_label = 'B-PAD'  # do not use the letter "O" because evaluation requires only removing padding

    train_data_path = RemoteDirs.data / 'CONLL05/conll05.train.txt'
    dev_data_path = RemoteDirs.data / 'CONLL05/conll05.dev.txt'
    test_data_path = RemoteDirs.data / 'CONLL05/conll05.test.wsj.txt'
    glove_path = RemoteDirs.data / 'glove.6B.100d.txt'
    glove_path_local = LocalDirs.data / 'glove.6B.100d.txt' if socket.gethostname() == 'Philum' else None

    verbose = True


class Eval:
    loss_interval = 100
    summary_interval = 100
    verbose = True  # print output of perl evaluation script
    dev_batch_size = 512  # too big will cause tensorflow internal error
    srl_eval_path = RemoteDirs.root / 'perl' / 'srl-eval.pl'



