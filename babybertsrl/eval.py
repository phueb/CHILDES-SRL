import torch
from allennlp.data.iterators import BucketIterator

from babybertsrl.scorer import SrlEvalScorer, convert_bio_tags_to_conll_format


def predict_masked_sentences(model, test_data, output_vocab):
    model.eval()

    # make test batch
    instances = test_data.make_instances(test_data.utterances)
    num_instances = len(test_data.utterances)
    bucket_batcher = BucketIterator(batch_size=num_instances, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(output_vocab)
    batch = next(bucket_batcher(instances, num_epochs=1))

    # get predictions
    with torch.no_grad():
        output_dict = model(**batch)  # input is dict[str, tensor]

    # show results only for whole-words
    lm_in = output_dict['lm_in']
    predicted_lm_out = model.decode(output_dict).pop("lm_tags")
    gold_lm_tags = output_dict['gold_lm_tags']
    assert len(lm_in) == len(predicted_lm_out) == len(gold_lm_tags)

    for a, b, c in zip(lm_in, predicted_lm_out, gold_lm_tags):
        print(len(a), len(b), len(c), flush=True)
        for ai, bi, ci in zip(a, b, c):
            print(f'{ai:>20} {bi:>20} {ci:>20}', flush=True)  # TODO save to file
    print(flush=True)


def evaluate_model_on_pp(model, instances_generator):
    model.eval()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for step, batch in enumerate(instances_generator):

        # get predictions
        with torch.no_grad():
            output_dict = model(**batch)  # input is dict[str, tensor]

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
        output_dict = model(**batch)  # input is dict[str, tensor]

        # metadata
        metadata = batch['metadata']
        batch_verb_indices = [example_metadata['verb_index'] for example_metadata in metadata]
        batch_sentences = [example_metadata['words'] for example_metadata in metadata]

        # Get the BIO tags from decode()
        batch_bio_predicted_tags = model.decode(output_dict).pop("tags")
        batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                      tags in batch_bio_predicted_tags]
        batch_bio_gold_tags = [example_metadata['gold_tags'] for example_metadata in metadata]
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