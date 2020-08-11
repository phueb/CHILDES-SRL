from pathlib import Path
from typing import Iterator
import numpy as np

import torch

from babybertsrl.model import MTBert


def score_forced_choices(model: MTBert,
                         instances_generator: Iterator,
                         out_path: Path,
                         verbose: bool = False):
    model.eval()

    mlm_in = []
    cross_entropies = []

    for batch in instances_generator:

        with torch.no_grad():
            output_dict = model(task='forced_choice', **batch)  # input is dict[str, tensor]
            # get cross_entropies + metadata
            mlm_in += output_dict['in']
            loss = output_dict['loss'].detach().cpu().numpy()  # shape is [batch size, seq length]
            attention_mask = output_dict['attention_mask'].detach().cpu().numpy()  # shape is [batch size, seq length]
            # we need 1 loss value per utterance.
            # to do so, we must exclude loss for padding symbols, using attention_mask provided by AllenNLP logic
            loss_cleaned = [row[np.where(row_mask)[0]].mean().item() for row, row_mask in zip(loss, attention_mask)]
            cross_entropies += loss_cleaned
            assert len(mlm_in) == len(cross_entropies)

    # save to file
    print(f'Saving forced_choice probing results to {out_path}')
    with out_path.open('w') as f:
        for s, xe in zip(mlm_in, cross_entropies):
            line = f'{" ".join(s)} {xe:.4f}'
            f.write(line + '\n')
            if verbose:
                print(line)


def predict_masked_sentences(model: MTBert,
                             instances_generator: Iterator,
                             out_path: Path,
                             print_gold: bool = True,
                             verbose: bool = False):
    model.eval()

    mlm_in = []
    predicted_mlm_tags = []
    gold_mlm_tags = []

    for batch in instances_generator:

        with torch.no_grad():
            output_dict = model(task='mlm', **batch)  # input is dict[str, tensor]
            # get predictions + metadata
            predicted_mlm_tags += model.decode_mlm(output_dict)
            mlm_in += output_dict['in']
            gold_mlm_tags += output_dict['gold_tags']

    # save to file
    print(f'Saving open_ended probing results to {out_path}')
    with out_path.open('w') as f:
        for a, b, c in zip(mlm_in, predicted_mlm_tags, gold_mlm_tags):
            assert len(a) == len(b)
            for ai, bi, ci in zip(a, b, c):  # careful, zips over shortest list
                if print_gold:
                    line = f'{ai:>20} {bi:>20} {ci:>20}'
                else:
                    line = f'{ai:>20} {bi:>20}'
                f.write(line + '\n')
                if verbose:
                    print(line)
            f.write('\n')
            if verbose:
                print('\n')