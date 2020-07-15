# BabyBERTSRL

## Background

This repository contains research code that trains Allen AI's implementation of a BERT-based SRL labeler. 
A demo can be found [here](https://demo.allennlp.org/semantic-role-labeling). 
The paper describing the original implementation (Shi et al., 2019) can be found [here](https://arxiv.org/abs/1904.05255).

The code is for research purpose only. 
The goal of this research project is to test theories of how children learn to understand multi-word utterances. 

## History

- 2008: The BabySRL project started as a collaboration between Cynthia Fisher, Dan Roth, Michael Connor and Yael Gertner, 
whose published work is available [here](https://www.aclweb.org/anthology/W08-2111/).

- 2016: The most recent work, prior to this, can be found [here](https://gitlab-beta.engr.illinois.edu/babysrl-group/babysrl)

- 2019: The current work is an extension of the previous, leveraging the powerful deep-learning model BERT. Investigation into the inner workings of the model began in in September 2019 while under the supervision of [Cynthia Fisher](https://psychology.illinois.edu/directory/profile/clfishe)
at the Department of Psychology at [UIUC](https://psychology.illinois.edu/). 

- 2020: Experimentation with with joint training BERT on SRL and MLM began. The joint training procedure is similar to what is proposed in [https://arxiv.org/pdf/1901.11504.pdf](https://arxiv.org/pdf/1901.11504.pdf)

## BERT
 
Due to the limited size of child-directed speech data, 
a much smaller BERT than the standard BERT models is trained here.

For example, compare the architecture specified in `params.py` to the
architecture of the state-of-the-art BERT-based SRL tagger [here](https://github.com/allenai/allennlp/blob/master/training_config/bert_base_srl.jsonnet)

Moreover, no next-sentence prediction objective is used during training, as was done in the original implementation. This reduces training time, code complexity and [learning two separate semantic spaces](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1117&context=scil).

## Working with the AllenNLP toolkit

* Utterances are loaded from text file
* A word in each utterance is masked
* Each utterance is converted to word-pieces using a custom vocab file
* Each utterance is converted to an instance
* A vocabulary for input and output words is created from train and test instances
* A Bert model is instantiated using the size of the vocabulary for input words
* A LM model is instantiated with the Bert model providing word embeddings
* The LM model adds a projection layer on top of Bert mapping from hidden size -> size of vocabulary for output words

Because the vocabulary holds word pieces for both input and output words, and the model works with word-pieces only,
a decoding function is called which converts the word pieces back into whole words.

## Custom Vocabulary

If using a custom vocab file, use the following formatting:
```text
 2342 .
 1249 you
  234 I
  .
  .
  .
```

There are two columns, one containing frequencies, the other words. 
The columns must be sorted in order of decreasing frequency (most frequent word at the top).
Place the file in `data`, and specify its name in `params.py`.

## Evaluating syntactic knowledge

Generate test sentences (with number agreement) and evaluate the model's predictions using [Babeval](https://github.com/phueb/Babeval).


## Compatibility

Tested on Ubuntu 16.04, Python 3.6, and torch==1.2.0
