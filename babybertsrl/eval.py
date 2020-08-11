import torch
from typing import Iterator, Optional
from pathlib import Path

from babybertsrl.scorer import SrlEvalScorer, convert_bio_tags_to_conll_format
from babybertsrl.model import MTBert


def evaluate_model_on_pp(model: MTBert,
                         instances_generator: Iterator,
                         ) -> float:
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


def evaluate_model_on_f1(model: MTBert,
                         srl_eval_path: Path,
                         instances_generator: Iterator,
                         save_path: Optional[Path] = None,
                         print_tag_metrics: bool = False,
                         ) -> float:

    scorer = SrlEvalScorer(srl_eval_path,
                           ignore_classes=['V'])

    model.eval()
    for step, batch in enumerate(instances_generator):

        # get predictions
        with torch.no_grad():
            output_dict = model(task='srl', **batch)  # input is dict[str, tensor]

        # metadata
        metadata = batch['metadata']
        batch_verb_indices = [example_metadata['verb_index'] for example_metadata in metadata]
        batch_sentences = [example_metadata['in'] for example_metadata in metadata]

        # Get the BIO tags from decode()
        batch_bio_predicted_tags = model.decode_srl(output_dict)
        batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                      tags in batch_bio_predicted_tags]
        batch_bio_gold_tags = [example_metadata['gold_tags'] for example_metadata in metadata]
        batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                                 tags in batch_bio_gold_tags]

        # update signal detection metrics
        scorer(batch_verb_indices,
               batch_sentences,
               batch_conll_predicted_tags,
               batch_conll_gold_tags)

    # compute f1 on accumulated signal detection metrics and reset
    tag2metrics = scorer.get_tag2metrics(reset=True)

    # print f1 summary by tag
    if print_tag_metrics:
        scorer.print_summary(tag2metrics)

    # save tag f1 dict to csv
    if save_path is not None:
        out_path = save_path / 'f1_by_tag.csv'
        scorer.save_tag2metrics(out_path, tag2metrics)

    return tag2metrics['overall']['f1']
