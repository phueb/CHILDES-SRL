import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

from pytorch_pretrained_bert import BertAdam

from babybertsrl.data import Data
from babybertsrl.model import make_bertsrl
from babybertsrl import config
from babybertsrl.eval import evaluate_model_on_f1


@attr.s
class Params(object):
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))

    max_sentence_length = attr.ib(validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(validator=attr.validators.instance_of(int))

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    #  paths
    project_path = Path(param2val['project_path'])
    srl_eval_path = project_path / 'perl' / 'srl-eval.pl'
    train_data_path = project_path / 'data' / 'CONLL05/conll05.train.txt'
    dev_data_path = project_path / 'data' / 'CONLL05/conll05.dev.txt'
    test_data_path = project_path / 'data' / 'CONLL05/conll05.test.wsj.txt'

    # data + vocab + batcher
    data = Data(params, train_data_path, dev_data_path)
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)
    vocab.print_statistics()
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)  # this must be an Allen Vocabulary instance
    print(f'Allen Vocab size={vocab.get_vocab_size("tokens")}')

    bert_vocab = data.bert_tokenizer.vocab
    vocab_size = len(bert_vocab)
    print(f'Bert Vocab size={vocab_size:,}')

    # model + optimizer
    model = make_bertsrl(params, vocab)  # this must be an Allen Vocabulary instance
    optimizer = BertAdam(params=model.parameters(),
                         lr=5e-5,
                         max_grad_norm=1.0,
                         t_total=-1,
                         weight_decay=0.01)

    # train + eval loop
    dev_f1s = []
    train_f1s = []
    train_start = time.time()
    for epoch in range(params.num_epochs):
        print(f'\nEpoch: {epoch}')

        # evaluate f1
        dev_f1 = evaluate_model_on_f1(model, params, srl_eval_path, bucket_batcher, data.dev_instances)
        # train_f1 = evaluate_model_on_f1(model, params, srl_eval_path, bucket_batcher, data.train_instances)
        train_f1 = None  # TODO takes long
        dev_f1s.append(dev_f1)
        train_f1s.append(train_f1)
        print(f'train-f1={train_f1}')
        print(f'dev-f1={dev_f1}')

        # train
        model.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):
            loss = model.train_on_batch(batch, optimizer)
            # print
            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    s1 = pd.Series(train_f1s, index=np.arange(params.num_epochs))
    s1.name = 'train_f1'
    s2 = pd.Series(dev_f1s, index=np.arange(params.num_epochs))
    s2.name = 'dev_f1'

    return [s1, s2]