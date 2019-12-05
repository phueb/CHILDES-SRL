import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch
from itertools import chain

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.util import move_optimizer_to_cuda

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert import BertAdam

from babybertsrl import config
from babybertsrl.data_lm import DataLM
from babybertsrl.data_srl import DataSRL
from babybertsrl.eval import evaluate_model_on_pp
from babybertsrl.eval import predict_masked_sentences
from babybertsrl.model_lm import LMBert
from babybertsrl.model_srl import SrlBert
from babybertsrl.eval import evaluate_model_on_f1


@attr.s
class Params(object):
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))
    max_sentence_length = attr.ib(validator=attr.validators.instance_of(int))
    num_pre_train_epochs = attr.ib(validator=attr.validators.instance_of(int))
    num_fine_tune_epochs = attr.ib(validator=attr.validators.instance_of(int))
    num_masked = attr.ib(validator=attr.validators.instance_of(int))
    vocab_size = attr.ib(validator=attr.validators.instance_of(int))
    corpus_name = attr.ib(validator=attr.validators.instance_of(str))

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
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
    train_data_path_lm = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_train_lm.txt'
    dev_data_path_lm = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_dev_lm.txt'
    test_data_path_lm = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_test_lm.txt'
    train_data_path_srl = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_train_srl.txt'
    dev_data_path_srl = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_dev_srl.txt'
    test_data_path_srl = project_path / 'data' / 'CHILDES' / f'{params.corpus_name}_test_srl.txt'
    vocab_path = project_path / 'data' / f'{params.corpus_name}_vocab_{params.vocab_size}.txt'

    # BERT tokenizer - defines input vocabulary
    bert_tokenizer = BertTokenizer(str(vocab_path),
                                   do_basic_tokenize=False,
                                   do_lower_case=False)  # set to false because [MASK] must be uppercase

    # load utterances for pre-training
    train_data_lm = DataLM(params, train_data_path_lm, bert_tokenizer)
    dev_data_lm = DataLM(params, dev_data_path_lm, bert_tokenizer)
    test_data_lm = DataLM(params, test_data_path_lm, bert_tokenizer)

    # load propositions for fine-tuning on SRL task
    # train_data_srl = DataSRL(params, train_data_path_srl, bert_tokenizer)
    # dev_data_srl = DataSRL(params, dev_data_path_srl, bert_tokenizer)
    # test_data_srl = DataSRL(params, test_data_path_srl, bert_tokenizer)

    # TODO merge lm and srl instances to get combined output vocab?

    # get indexing_vocab
    # note: Allen NLP vocab holds labels, bert_tokenizer.vocab holds input tokens
    # what from_instances() does:
    # 1. it iterates over all instances, and all fields, and all token indexers
    # 2. the token indexer is used to update vocabulary count, skipping words whose text_id is already set
    all_instances = chain(train_data_lm.instances, dev_data_lm.instances)
    indexing_vocab = Vocabulary.from_instances(all_instances)
    indexing_vocab.print_statistics()

    # batcher
    bucket_batcher = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(indexing_vocab)  # sets vocab as attribute of bucket_batcher

    # BERT  # TODO original implementation used slanted_triangular learning rate scheduler
    # parameters of original implementation are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/bert_base_srl.jsonnet
    print('Preparing BERT for pre-training...')
    input_vocab_size = len(train_data_lm.bert_tokenizer.vocab)
    bert_config = BertConfig(vocab_size_or_config_json_file=input_vocab_size,  # was 32K
                             hidden_size=params.hidden_size,  # was 768
                             num_hidden_layers=params.num_layers,  # was 12
                             num_attention_heads=params.num_attention_heads,  # was 12
                             intermediate_size=params.intermediate_size)  # was 3072
    bert_model = BertModel(config=bert_config)

    # TODO mismatch between vocabs?

    print(input_vocab_size)
    print(indexing_vocab.get_vocab_size('tokens'))
    raise SystemExit

    # BERT + LM head
    bert_lm = LMBert(vocab=indexing_vocab,
                     bert_model=bert_model,
                     embedding_dropout=0.1)
    bert_lm.cuda()
    num_params = sum(p.numel() for p in bert_lm.parameters() if p.requires_grad)
    print('Number of model parameters: {:,}'.format(num_params), flush=True)
    optimizer_lm = BertAdam(params=bert_lm.parameters(),
                            lr=5e-5,
                            max_grad_norm=1.0,
                            t_total=-1,
                            weight_decay=0.01)
    move_optimizer_to_cuda(optimizer_lm)

    # ///////////////////////////////////////////
    # pre train
    # ///////////////////////////////////////////

    predict_masked_sentences(bert_lm, test_data_lm, indexing_vocab)

    dev_pps = []
    train_pps = []
    eval_steps = []
    train_start = time.time()
    for epoch in range(params.num_pre_train_epochs):
        print(f'\nEpoch: {epoch}', flush=True)

        # evaluate train perplexity
        train_pp = None
        train_pps.append(train_pp)
        print(f'train-pp={train_pp}', flush=True)

        # train
        bert_lm.train()
        train_generator = bucket_batcher(train_data_lm.make_instances(train_data_lm.utterances), num_epochs=1)
        for step, batch in enumerate(train_generator):
            loss = bert_lm.train_on_batch(batch, optimizer_lm)

            if step % config.Eval.loss_interval == 0:

                # evaluate perplexity
                instances_generator = bucket_batcher(dev_data_lm.make_instances(dev_data_lm.utterances), num_epochs=1)
                dev_pp = evaluate_model_on_pp(bert_lm, instances_generator)
                dev_pps.append(dev_pp)
                eval_steps.append(step)
                print(f'dev-pp={dev_pp}', flush=True)

                # test sentences
                predict_masked_sentences(bert_lm, test_data_lm, indexing_vocab)  # TODO save results to file

                # console
                min_elapsed = (time.time() - train_start) // 60
                print(f'step {step:<6}: pp={torch.exp(loss):2.4f} total minutes elapsed={min_elapsed:<3}', flush=True)

    # to pandas
    s1 = pd.Series(train_pps, index=np.arange(params.num_pre_train_epochs))
    s1.name = 'train_pp'
    s2 = pd.Series(dev_pps, index=eval_steps)
    s2.name = 'dev_pp'

    # ///////////////////////////////////////////
    # fine-tune
    # ///////////////////////////////////////////

    print('Preparing BERT for fine-tuning...')
    bert_srl = SrlBert(vocab=indexing_vocab,
                       bert_model=bert_model,  # bert_model is reused 
                       embedding_dropout=0.1)
    bert_srl.cuda()

    num_params = sum(p.numel() for p in bert_srl.parameters() if p.requires_grad)
    print('Number of model parameters: {:,}'.format(num_params), flush=True)
    optimizer_srl = BertAdam(params=bert_srl.parameters(),
                             lr=5e-5,
                             max_grad_norm=1.0,
                             t_total=-1,
                             weight_decay=0.01)

    dev_f1s = []
    train_f1s = []
    eval_steps = []
    train_start = time.time()
    for epoch in range(params.num_fine_tune_epochs):
        print(f'\nEpoch: {epoch}', flush=True)

        # evaluate train f1
        train_f1 = None
        train_f1s.append(train_f1)
        print(f'train-f1={train_f1}', flush=True)

        # train
        bert_srl.train()
        train_generator = bucket_batcher(train_data_srl.make_instances(train_data_srl.propositions),
                                         num_epochs=1)
        for step, batch in enumerate(train_generator):
            loss = bert_srl.train_on_batch(batch, optimizer_srl)

            if step % config.Eval.loss_interval == 0:
                # evaluate dev f1
                instances_generator = bucket_batcher(dev_data_srl.make_instances(dev_data_srl.propositions),
                                                     num_epochs=1)
                dev_f1 = evaluate_model_on_f1(bert_srl, srl_eval_path, instances_generator)
                dev_f1s.append(dev_f1)
                eval_steps.append(step)
                print(f'dev-f1={dev_f1}', flush=True)

                # console
                min_elapsed = (time.time() - train_start) // 60
                print(f'step {step:<6}: loss={loss:2.4f} total minutes elapsed={min_elapsed:<3}', flush=True)

    # to pandas
    s3 = pd.Series(train_f1s, index=np.arange(params.num_fine_tune_epochs))
    s3.name = 'train_f1'
    s4 = pd.Series(dev_f1s, index=eval_steps)
    s4.name = 'dev_f1'

    return [s1, s2, s3, s4]