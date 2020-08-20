import torch
from typing import Iterator, Optional
from pathlib import Path

from allennlp.data.iterators import BucketIterator

from babybertsrl import configs
from babybertsrl.io import load_utterances_from_file
from babybertsrl.probing import predict_forced_choice, predict_open_ended
from babybertsrl.converter import ConverterMLM
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


def get_probing_predictions(probing_path: Path,
                            converter_mlm: ConverterMLM,
                            bucket_batcher_mlm_large: BucketIterator,
                            save_path: Path,
                            mt_bert: MTBert,
                            step: Optional[int] = None):

    if step is None:
        step = 'last-step'

    for probing_task_name in configs.Eval.probing_names:
        for task_type in ['forced_choice', 'open_ended']:
            # prepare data - data is expected to be located on shared drive
            probing_data_path_mlm = probing_path / task_type / f'{probing_task_name}.txt'
            if not probing_data_path_mlm.exists():
                print(f'WARNING: {probing_data_path_mlm} does not exist', flush=True)
                continue
            print(f'Starting probing with task={probing_task_name}', flush=True)
            probing_utterances_mlm = load_utterances_from_file(probing_data_path_mlm)
            probing_instances_mlm = converter_mlm.make_probing_instances(probing_utterances_mlm)
            probing_generator_mlm = bucket_batcher_mlm_large(probing_instances_mlm, num_epochs=1)
            # prepare output path
            probing_results_path = save_path / task_type / f'probing_{probing_task_name}_results_{step}.txt'
            if not probing_results_path.parent.exists():
                probing_results_path.parent.mkdir(exist_ok=True)
            # inference + save results to file for scoring offline
            if task_type == 'forced_choice':
                predict_forced_choice(mt_bert,
                                      probing_generator_mlm,
                                      probing_results_path,
                                      verbose=True if 'dummy' in probing_task_name else False)
            elif task_type == 'open_ended':
                predict_open_ended(mt_bert,
                                   probing_generator_mlm,
                                   probing_results_path,
                                   print_gold=False,
                                   verbose=True if 'dummy' in probing_task_name else False)
            else:
                raise AttributeError('Invalid arg to "task_type".')