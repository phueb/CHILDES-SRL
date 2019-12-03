from spacy.tokens import Doc

from allennlp.predictors.predictor import Predictor

from babybertsrl.data_lm import Data
from babybertsrl import config
from babybertsrl.job import Params
from babybertsrl.params import param2default

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
                                cuda_device=0)

# data
data_path = config.Dirs.data / 'CHILDES' / 'childes-20191202_terms.txt'
params = Params.from_param2val(param2default)
data = Data(params, data_path, vocab_path_name=None)

for tokenized_utterance in data.utterances:
    # tag verbs
    spacy_doc = Doc(predictor._tokenizer.spacy.vocab, words=tokenized_utterance)
    for pipe in filter(None, predictor._tokenizer.spacy.pipeline):
        pipe[1](spacy_doc)  # this does POS tagging

    # to instances ( one for each verb in utterance
    tokens = [token for token in spacy_doc]
    words = [token.text for token in tokens]
    instances = []
    for i, word in enumerate(tokens):
        print(word.pos_)
        if word.pos_ == "VERB":
            verb_labels = [0 for _ in words]
            verb_labels[i] = 1
            instance = predictor._dataset_reader.text_to_instance(tokens, verb_labels)
            instances.append(instance)
    # in case no verbs are found
    if not instances:
        continue
    #
    res = predictor.predict_instances(instances)
    #
    print(tokens)
    print(res)
    print()
