"""
How well does AllenNLP SRL tagger perform on CHILDES 2008 SRL data?

          ARG-A1 f1= 0.00
          ARG-A4 f1= 0.00
         ARG-LOC f1= 0.00
            ARG0 f1= 0.95
            ARG1 f1= 0.93
            ARG2 f1= 0.79
            ARG3 f1= 0.44
            ARG4 f1= 0.80
        ARGM-ADV f1= 0.70
        ARGM-CAU f1= 0.84
        ARGM-COM f1= 0.00
        ARGM-DIR f1= 0.48
        ARGM-DIS f1= 0.68
        ARGM-EXT f1= 0.38
        ARGM-GOL f1= 0.00
        ARGM-LOC f1= 0.68
        ARGM-MNR f1= 0.68
        ARGM-MOD f1= 0.78
        ARGM-NEG f1= 0.99
        ARGM-PNC f1= 0.03
        ARGM-PPR f1= 0.00
        ARGM-PRD f1= 0.15
        ARGM-PRP f1= 0.39
        ARGM-RCL f1= 0.00
        ARGM-REC f1= 0.00
        ARGM-TMP f1= 0.84
          ARGRG1 f1= 0.00
          R-ARG0 f1= 0.00
          R-ARG1 f1= 0.00
      R-ARGM-CAU f1= 0.00
      R-ARGM-LOC f1= 0.00
      R-ARGM-TMP f1= 0.00
         overall f1= 0.88
"""

from spacy.tokens import Doc
from typing import Generator, List
from pathlib import Path

from allennlp.predictors.predictor import Predictor
from allennlp.data.instance import Instance

from babybertsrl import config
from babybertsrl.scorer import SrlEvalScorer, convert_bio_tags_to_conll_format

CORPUS_NAME = 'human-based-2008'
INTERACTIVE = False
BATCH_SIZE = 128


def gen_instances_from_gold(verb_index: int,
                            words: List[str],
                            gold_tags: List[str],
                            ) -> Generator[Instance, None, None]:
    # POS-tagging
    spacy_doc = Doc(predictor._tokenizer.spacy.vocab, words=words)
    for pipe in filter(None, predictor._tokenizer.spacy.pipeline):
        pipe[1](spacy_doc)  # this does POS tagging

    # to instances - one for each verb in utterance
    tokens = [token for token in spacy_doc]
    for i, word in enumerate(tokens):
        if word.pos_ == "VERB":

            if i != verb_index:  # only evaluate predictions for propositions with same verb index
                continue

            # instance
            verb_labels = [0 for _ in words]
            verb_labels[i] = 1
            instance = predictor._dataset_reader.text_to_instance(tokens, verb_labels)

            # meta data
            metadata = dict()
            metadata['in'] = words
            metadata['verb_index'] = i
            metadata['gold_tags'] = gold_tags

            yield instance, metadata


def gen_instances(file_path: Path) -> Generator[Instance, None, None]:
    with file_path.open('r') as f:
        for line in f.readlines():
            inputs = line.strip().split('|||')
            left_input = inputs[0].strip().split()
            right_input = inputs[1].strip().split()
            verb_index = int(left_input[0])
            words = left_input[1:]
            gold_tags = right_input

            instances = gen_instances_from_gold(verb_index, words, gold_tags)
            # in case no verbs are found
            if not instances:
                continue
            else:
                yield from instances


# srl tagger
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                                cuda_device=0)

# scorer
srl_eval_path = config.Dirs.root / 'perl' / 'srl-eval.pl'
span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=['V'])


gold_path = config.Dirs.data / 'training' / f'{CORPUS_NAME}_srl.txt'
for n, (instance, md) in enumerate(gen_instances(gold_path)):

    # get SRL predictions (decoding included)
    output_dict = predictor._model.forward_on_instance(instance)

    # metadata
    batch_verb_indices = [md['verb_index']]
    batch_sentences = [md['in']]

    # convert to conll
    batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(output_dict['tags'])]
    batch_conll_gold_tags = [convert_bio_tags_to_conll_format(md['gold_tags'])]

    # console
    print(batch_sentences)
    print(batch_conll_predicted_tags)
    print(batch_conll_gold_tags)

    # update signal detection metrics
    span_metric(batch_verb_indices,
                batch_sentences,
                batch_conll_predicted_tags,
                batch_conll_gold_tags)

    f1 = span_metric.get_metric(reset=False)['f1-measure-overall']
    print(f'proposition={n:>6,} f1={f1:.4f}')
    print()

# compute f1 on accumulated signal detection metrics and reset
metric_dict = span_metric.get_metric(reset=True)

# print f1 summary by tag
for k, v in sorted(metric_dict.items()):
    if k.startswith('f1-measure-'):
        tag = k.lstrip('f1-measure-')
        print(f"{tag:>16} f1={v: .2f}")
