from spacy.tokens import Doc
import pyprind

from allennlp.predictors.predictor import Predictor

from babybertsrl.io import load_utterances_from_file
from babybertsrl import config
from babybertsrl.job import Params
from babybertsrl.params import param2default

CORPUS_NAME = 'childes-20191204'
INSPECT_ONLY = False
INTERACTIVE = False

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                                cuda_device=0)

# utterances
utterances_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_mlm.txt'
params = Params.from_param2val(param2default)
utterances = load_utterances_from_file(utterances_path)

# open files
srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_srl.txt'
out_f = srl_path.open('w')

progress_bar = pyprind.ProgBar(len(utterances),
                              title=f'Srl tagging..',
                              stream=1)
for tokenized_utterance in utterances:

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
            out_f.close()
            raise SystemExit('Quit')

    # write to file
    left_input = ' '.join(tokenized_utterance)
    for d in res['verbs']:
        # make line
        tags = d['tags']

        # TODO debugging
        print(tags)

        verb_index = tags.index('B-V')
        right_input = ' '.join(tags)
        line = f'{verb_index} {left_input} ||| {right_input}'

        if INTERACTIVE:
            print(line)
            key = input('\n[q] to quit. Any key to continue.\n')
            if key != 'q':
                pass
            else:
                out_f.close()
                raise SystemExit('Quit')

        out_f.write(line + '\n')

    progress_bar.update()

out_f.close()