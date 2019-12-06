from spacy.tokens import Doc
import pyprind
from deepsegment import DeepSegment

from allennlp.predictors.predictor import Predictor

from babybertsrl.io import load_utterances_from_file
from babybertsrl import config
from babybertsrl.job import Params
from babybertsrl.params import param2default

CORPUS_NAME = 'childes-20191206'
INTERACTIVE = False

# srl tagger
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                                cuda_device=0)

# segmenter for splitting ill-formed utterances into well-formed sentences
segmenter = DeepSegment('en')

# utterances
utterances_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_mlm.txt'
params = Params.from_param2val(param2default)
utterances = load_utterances_from_file(utterances_path)

progress_bar = pyprind.ProgBar(len(utterances), stream=1)
num_no_verb = 0
lines = []
for unsegmented_words in utterances:

    # possibly segment utterance into multiple well-formed sentences
    words_string = ' '.join(unsegmented_words)
    segments = segmenter.segment(words_string)

    for segment in segments:

        # tag verbs
        words = segment.split()
        spacy_doc = Doc(predictor._tokenizer.spacy.vocab, words=words)
        for pipe in filter(None, predictor._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)  # this does POS tagging

        # to instances ( one for each verb in utterance
        tokens = [token for token in spacy_doc]
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

        # get SRL predictions
        # print(f'Predicting SRL tags for {len(instances)} instances')
        res = predictor.predict_instances(instances)

        # write to file
        for d in res['verbs']:

            # sometimes there is no B-V
            if 'B-V' not in d['tags']:
                num_no_verb += 1
                continue

            # make line
            tags = d['tags']
            verb_index = tags.index('B-V')
            line = f'{verb_index} {segment} ||| {" ".join(tags)}'

            if INTERACTIVE:
                print('=====================================')
                print(d['description'])
                print(line)
                key = input('\n[q] to quit. Any key to continue.\n')
                if key != 'q':
                    pass
                else:
                    raise SystemExit('Quit')

            lines.append(line)

    progress_bar.update()

print(f'Skipped {num_no_verb} utterances due to absence of B-V tag')

print('Writing to file...')
srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_srl.txt'
with srl_path.open('w') as f:
    for line in lines:
        f.write(line + '\n')

