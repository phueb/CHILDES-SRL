import torch
from typing import Iterator

from allennlp.models import Model

from babybertsrl.scorer import SrlEvalScorer, convert_bio_tags_to_conll_format


def predict_masked_sentences(model: Model,
                             instances_generator: Iterator,
                             verbose: bool = False):
    model.eval()

    # get batch
    batch = next(instances_generator)

    # TODO currently only first batch in test split is considered

    # get predictions
    with torch.no_grad():
        output_dict = model(task='mlm', **batch)  # input is dict[str, tensor]

    # show results only for whole-words
    mlm_in = output_dict['mlm_in']
    predicted_mlm_tags = model.decode(output_dict)
    gold_mlm_tags = output_dict['gold_mlm_tags']
    assert len(mlm_in) == len(predicted_mlm_tags) == len(gold_mlm_tags)

    # TODO save to file
    if verbose:
        for a, b, c in zip(mlm_in, predicted_mlm_tags, gold_mlm_tags):
            print(len(a), len(b), len(c), flush=True)
            for ai, bi, ci in zip(a, b, c):
                print(f'{ai:>20} {bi:>20} {ci:>20}', flush=True)

        print(flush=True)


def evaluate_model_on_pp(model, instances_generator):
    model.eval()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for step, batch in enumerate(instances_generator):

        # get predictions
        with torch.no_grad():
            output_dict = model(task='mlm', **batch)  # input is dict[str, tensor]

        pp = torch.exp(output_dict['loss'])
        pp_sum += pp
        num_steps += 1

    return pp_sum.cpu().numpy().item() / num_steps


def evaluate_model_on_f1(model, srl_eval_path, instances_generator):

    span_metric = SrlEvalScorer(srl_eval_path,
                                ignore_classes=['V'])

    model.eval()
    for step, batch in enumerate(instances_generator):

        # get predictions
        output_dict = model(task='srl', **batch)  # input is dict[str, tensor]

        # metadata
        metadata = batch['metadata']
        batch_verb_indices = [example_metadata['verb_index'] for example_metadata in metadata]
        batch_sentences = [example_metadata['srl_in'] for example_metadata in metadata]

        # Get the BIO tags from decode()
        batch_bio_predicted_tags = model.decode(output_dict, task='srl')
        batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                      tags in batch_bio_predicted_tags]
        batch_bio_gold_tags = [example_metadata['gold_srl_tags'] for example_metadata in metadata]
        batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                                 tags in batch_bio_gold_tags]

        # update signal detection metrics
        span_metric(batch_verb_indices,
                    batch_sentences,
                    batch_conll_predicted_tags,
                    batch_conll_gold_tags)

    # compute f1 on accumulated signal detection metrics and reset
    metric_dict = span_metric.get_metric(reset=True)
    return metric_dict['f1-measure-overall']