from spacy.tokens import Doc
import random

from allennlp.predictors.predictor import Predictor

from babybertsrl.data_lm import DataLM
from babybertsrl import config
from babybertsrl.job import Params
from babybertsrl.params import param2default

CORPUS_NAME = 'childes-20191204'
DEVEL_PROB = 0.5
INSPECT_ONLY = False
INTERACTIVE = True

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                                cuda_device=0)

# data
data_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}.txt'
params = Params.from_param2val(param2default)
data = DataLM(params, data_path, bert_tokenizer=None)

# open files
train_srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_train_srl.txt'
devel_srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_devel_srl.txt'
train_srl_f = train_srl_path.open('w')
devel_srl_f = devel_srl_path.open('w')

for tokenized_utterance in data.utterances:

    # TODO use spacy sentence boundary detection before srl tagging ?

    # tag verbs
    spacy_doc = Doc(predictor._tokenizer.spacy.vocab, words=tokenized_utterance)
    for pipe in filter(None, predictor._tokenizer.spacy.pipeline):
        pipe[1](spacy_doc)  # this does POS tagging

    # to instances ( one for each verb in utterance
    tokens = [token for token in spacy_doc]
    words = [token.text for token in tokens]
    instances = []
    for i, word in enumerate(tokens):
        if word.pos_ == "VERB":
            verb_labels = [0 for _ in words]
            verb_labels[i] = 1
            instance = predictor._dataset_reader.text_to_instance(tokens, verb_labels)
            instances.append(instance)
    # in case no verbs are found
    if not instances:
        continue

    res = predictor.predict_instances(instances)

    if INSPECT_ONLY:
        for d in res['verbs']:
            print(d['verb'])
            print(d['description'])
        print()

        key = input('[q] to quit. Any key to continue.')
        if key != 'q':
            continue
        else:
            train_srl_f.close()
            devel_srl_f.close()
            raise SystemExit('Quit')

    # write to file
    left_input = ' '.join(tokenized_utterance)
    for d in res['verbs']:
        # make line
        tags = d['tags']
        verb_index = tags.index('B-V')
        right_input = ' '.join(tags)
        line = f'{verb_index} {left_input} ||| {right_input}'

        if INTERACTIVE:
            print(line)
            key = input('\n[q] to quit. Any key to continue.\n')
            if key != 'q':
                pass
            else:
                train_srl_f.close()
                devel_srl_f.close()
                raise SystemExit('Quit')

        # if True, write line to devel_srl
        if random.choices([True, False], weights=[DEVEL_PROB, 1 - DEVEL_PROB])[0]:
            devel_srl_f.write(line + '\n')
        else:
            train_srl_f.write(line + '\n')

train_srl_f.close()
devel_srl_f.close()