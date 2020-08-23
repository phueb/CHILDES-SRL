"""
WARNING: Not intended to be used as-is.
This file is a suggested recipe for training BERT jointly on MLM and SRL.

"""
from typing import Dict, Any, Optional, List
import time
import numpy as np
import torch
import random
import attr
from itertools import cycle

from childes_srl import configs
from childes_srl.io import load_mlm_data
from childes_srl.io import load_srl_data
from bert_recipes.model import BertForMLMAndSRL
from bert_recipes.eval import evaluate_model_on_f1
from bert_recipes.decode import decode_mlm_batch_output


@attr.s
class Params(object):
    """
    num_mlm_epochs: number of times to re-visit MLM examples during training
    srl_probability: probability of training on SRL batch after training on MLM batch
    srl_interleaved: True if training jointly on SRL and MLM
    """
    num_mlm_epochs = attr.ib(validator=attr.validators.instance_of(int))
    srl_probability = attr.ib(validator=attr.validators.instance_of(float))
    srl_interleaved = attr.ib(validator=attr.validators.instance_of(bool))

    @classmethod
    def from_dict(cls,
                  d: Dict[str, Any],
                  excluded: Optional[List[str]] = None,
                  ):

        if excluded is None:
            excluded = []

        kwargs = {k: v for k, v in d.items()
                  if k not in excluded}
        return cls(**kwargs)


def main(params: Params):

    # load data
    path_to_mlm_data = configs.Dirs.data / 'pre_processed' / f'childes-20191206_mlm.txt'
    path_to_srl_data = configs.Dirs.data / 'pre_processed' / f'childes-20191206_no-dev_srl.txt'
    data_mlm = load_mlm_data(path_to_mlm_data)
    data_srl = load_srl_data(path_to_srl_data)

    def to_batches(data: List[Any]) -> List[Any]:
        raise NotImplementedError

    def to_tensors(batch: List[Any]) -> Dict[str, torch.Tensor]:
        res = {'task': None,
               'input_ids': None,
               'token_type_ids': None,
               'attention_mask': None,
               'tags': None}
        raise NotImplementedError

    def to_meta_data(batch: List[Any]) -> Dict[str, Any]:
        res = {
            'tokens': tokens,  # for decoding MLM tags
            "attention_mask": attention_mask,  # for decoding BIO SRL tags
            'start_offsets': [],  # for decoding BIO SRL tags
            'in': [],  # for decoding MLM tags
            'gold_tags': [],  # for computing f1 score
        }
        raise NotImplementedError

    # make generators yielding tuples like (dict with tensors for training, dict with metadata for decoding)
    batches_mlm = ((to_tensors(batch), to_meta_data(batch)) for batch in to_batches(data_mlm))
    batches_srl = ((to_tensors(batch), to_meta_data(batch)) for batch in to_batches(data_srl))

    batches_srl = cycle(batches_srl)  # should be infinite generator

    bert_encoder = NotImplementedError  # TODO implement, e.g. hugginface transformers.BertModel
    model = BertForMLMAndSRL(bert_encoder,
                             num_tags_mlm,
                             num_tags_srl,
                             ignore_token_id,
                             )

    # max step does not take into consideration number of unique SRL batches because it does not vary with num_masked.
    # the SRL batcher is infinite, and yields a batch with probability = srl_probability when interleaved = True,
    # or stop when max_step is reached when interleaved = False
    num_train_mlm_batches = len(list(batches_mlm))
    max_step = num_train_mlm_batches + (params.srl_probability * num_train_mlm_batches)
    print(f'Will stop training at global step={max_step:,}')
    print(flush=True)

    # init
    evaluated_steps_srl = []
    evaluated_steps_mlm = []
    train_start = time.time()
    train_f1 = None
    loss_mlm = None
    no_mlm_batches = False
    step_mlm = 0
    step_srl = 0
    step_global = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    while step_global < max_step:

        # ####################################################################### TRAINING

        if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
            model.train()

            # masked language modeling objective
            try:
                batch_mlm, _ = next(batches_mlm)
            except StopIteration:
                if params.srl_interleaved:
                    break
                else:
                    no_mlm_batches = True
            else:
                # forward
                output_mlm = model(**batch_mlm)
                loss_mlm = output_mlm['loss']

                # backward + scale gradient + optimizer step
                loss_mlm.backward()
                pass  # TODO implement

                step_mlm += 1

            # semantic role labeling objective
            if (params.srl_interleaved and random.random() < params.srl_probability) or no_mlm_batches:
                batch_srl, _ = next(batches_srl)
                output_srl = model(**batch_srl)
                loss_srl = output_srl['loss']

                # backward + scale gradient + optimizer step
                loss_srl.backward()
                pass  # TODO implement

                step_srl += 1

        is_first_time_in_loop = False
        step_global = step_mlm + step_srl

        # ####################################################################### EVALUATION

        # eval MLM
        if step_mlm % configs.Example.eval_interval == 0 and step_mlm not in evaluated_steps_mlm:
            evaluated_steps_mlm.append(step_mlm)
            is_evaluated_at_current_step = True
            model.eval()

            # print out some MLM examples
            filled_in_utterances = decode_mlm_batch_output(token_ids,
                                                           logits,
                                                           utterances,
                                                           mask_token_id)
            for u in filled_in_utterances:
                print(u)

        # eval SRL
        if step_srl % configs.Example.eval_interval == 0 and step_srl not in evaluated_steps_srl:
            evaluated_steps_srl.append(step_srl)
            is_evaluated_at_current_step = True
            model.eval()

            # evaluate f1 on train data
            srl_eval_path = configs.Dirs.perl / 'srl-eval.pl'  # path to official perl script for scoring SRL
            train_f1 = evaluate_model_on_f1(model, srl_eval_path, batches_srl)
            print(f'train-f1={train_f1}', flush=True)

        # console
        if is_evaluated_at_current_step or step_global % configs.Example.feedback_interval == 0:
            min_elapsed = (time.time() - train_start) // 60
            pp = torch.exp(loss_mlm) if loss_mlm is not None else np.nan
            print(f'step MLM={step_mlm:>9,} | step SRL={step_srl:>9,} | step global={step_global:>9,}\n'
                  f'pp={pp :2.4f} \n'
                  f'total minutes elapsed={min_elapsed:<3}\n', flush=True)
            is_evaluated_at_current_step = False

    return train_f1


if __name__ == '__main':

    param2val = {'num_mlm_epochs': 1,
                 'srl_probability': 1.0,
                 'srl_interleaved': True,
                 }

    train_f1 = main(Params.from_dict(param2val))
    print(f'Finished training. End-of-training f1 on train data = {train_f1}')