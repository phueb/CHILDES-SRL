import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

from pytorch_pretrained_bert import BertAdam

from babybertsrl import config
from babybertsrl.data_lm import Data
from babybertsrl.eval import evaluate_model_on_pp
from babybertsrl.model_lm import make_bert_lm


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
    train_data_path = project_path / 'data' / 'CHILDES' / 'childes-20180319_train.txt'
    dev_data_path = project_path / 'data' / 'CHILDES' / 'childes-20180319_dev.txt'

    # data + vocab + batcher
    data = Data(params, train_data_path, dev_data_path)
    vocab = Vocabulary.from_instances(data.train_instances + data.dev_instances)
    vocab.print_statistics()
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)  # this must be an Allen Vocabulary instance

    # note:
    # the Vocab object has word-piece tokenized tokens, ready to be fed directly to bert.
    # the Vocab object uses a pre-made 30k bert vocabulary with which to build the vocabulary.
    # this means that words in the data not in the pre-made vocabulary are excluded
    print(f'Vocab size={vocab.get_vocab_size("tokens")}')

    # model + optimizer
    bert_lm = make_bert_lm(params, vocab)
    optimizer = BertAdam(params=bert_lm.parameters(),
                         lr=5e-5,
                         max_grad_norm=1.0,
                         t_total=-1,
                         weight_decay=0.01)

    # train + eval loop
    dev_pps = []
    train_pps = []
    train_start = time.time()
    for epoch in range(params.num_epochs):
        print(f'\nEpoch: {epoch}')

        # evaluate perplexity
        dev_pp = evaluate_model_on_pp(bert_lm, params, bucket_batcher, data.dev_instances)
        train_pp = evaluate_model_on_pp(bert_lm, params, bucket_batcher, data.train_instances)
        dev_pps.append(dev_pp)
        train_pps.append(train_pp)
        print(f'train-pp={train_pp}')
        print(f'dev-pp={dev_pp}')

        # train
        bert_lm.train()
        train_generator = bucket_batcher(data.train_instances, num_epochs=1)
        for step, batch in enumerate(train_generator):
            loss = bert_lm.train_on_batch(batch, optimizer)
            # print
            if step % config.Eval.loss_interval == 0:
                print('step {:<6}: loss={:2.2f} total minutes elapsed={:<3}'.format(
                    step, loss, (time.time() - train_start) // 60))

    # to pandas
    s1 = pd.Series(train_pps, index=np.arange(params.num_epochs))
    s1.name = 'train_pp'
    s2 = pd.Series(dev_pps, index=np.arange(params.num_epochs))
    s2.name = 'dev_pp'

    return [s1, s2]