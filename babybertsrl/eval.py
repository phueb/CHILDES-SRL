import torch
from allennlp.data.iterators import BucketIterator

from babybertsrl.scorer import SrlEvalScorer, convert_bio_tags_to_conll_format


def predict_masked_sentences(model, data, vocab):
    model.eval()

    # make test batch
    utterances = [['say', 'thank', 'you'],
                  ['what', "'s", 'that', '?'],
                  ['what', 'are', 'they', 'doing', '?']]
    instances = data.make_instances(utterances)
    num_instances = len(instances)
    bucket_batcher = BucketIterator(batch_size=num_instances, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher.index_with(vocab)  # this mus
    batch = next(bucket_batcher(instances, num_epochs=1))

    # get predictions
    with torch.no_grad():
        output_dict = model(**batch)  # input is dict[str, tensor]

    gold_lm_tags = output_dict['words']  # whole words
    masked_indices = output_dict['masked_indices']  # whole words
    predicted_lm_tags = model.decode(output_dict).pop("lm_tags")  # whole words
    assert len(gold_lm_tags) == len(predicted_lm_tags)

    for g, p, m in zip(gold_lm_tags, predicted_lm_tags, masked_indices):
        print(len(g), len(p), len(m))
        for gi, pi, mi in zip(g, p, m):
            print(f'{gi:>20} {pi:>20} {"masked" if mi else ""}')  # TODO save to file
    print()


def evaluate_model_on_pp(model, params, bucket_batcher, instances):
    model.eval()

    instances_generator = bucket_batcher(instances, num_epochs=1)
    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for step, batch in enumerate(instances_generator):

        if len(batch['lm_tags']) != params.batch_size:
            print('WARNING: Batch size is {}. Skipping'.format(len(batch['lm_tags'])))
            continue

        # get predictions
        with torch.no_grad():
            output_dict = model(**batch)  # input is dict[str, tensor]

        pp = torch.exp(output_dict['loss'])
        pp_sum += pp
        num_steps += 1

    return pp_sum.cpu().numpy().item() / num_steps


def evaluate_model_on_f1(model, params, srl_eval_path, bucket_batcher, instances):

    span_metric = SrlEvalScorer(srl_eval_path,
                                ignore_classes=["V"])

    model.eval()
    instances_generator = bucket_batcher(instances, num_epochs=1)
    for step, batch in enumerate(instances_generator):

        if len(batch['tags']) != params.batch_size:
            print('WARNING: Batch size is {}. Skipping'.format(len(batch['tags'])))
            continue

        # get predictions
        output_dict = model(**batch)  # input is dict[str, tensor]

        # metadata
        metadata = batch['metadata']
        batch_verb_indices = [example_metadata["verb_index"] for example_metadata in metadata]
        batch_sentences = [example_metadata["words"] for example_metadata in metadata]

        # Get the BIO tags from decode()
        batch_bio_predicted_tags = model.decode(output_dict).pop("tags")
        batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                      tags in batch_bio_predicted_tags]
        batch_bio_gold_tags = [example_metadata["gold_tags"] for example_metadata in metadata]
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