# BabyBERTSRL

## Background

This repository contains research code that trains Allen AI's implementation of a BERT-based SRL labeler. 
A demo can be found [here](https://demo.allennlp.org/semantic-role-labeling). 
The paper describing the original implementation (Shi et al., 2019) can be found [here](https://arxiv.org/abs/1904.05255).

The code is for research purpose only. 
The goal of this research project is to test theories of how children learn to understand multi-word utterances. 

## History

Investigation into the inner workings of the model began in in September 2019 while under the supervision of [Cynthia Fisher](https://psychology.illinois.edu/directory/profile/clfishe)
at the Department of Psychology at [UIUC](https://psychology.illinois.edu/). 

## BERT
 
Due to the limited size of child-directed speech data, 
a much smaller BERT than the standard BERT models is trained here.

For example, compare the architecture specified in `params.py` to the
architecture of the state-of-the-art BERT-based SRL tagger [here](https://github.com/allenai/allennlp/blob/master/training_config/bert_base_srl.jsonnet)

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

Currently, word-pieces are not supported. 
The word-piece vocab file must contain whole words only.
Word-piece tokenization therefore is a no-op.
Using word-pieces would require handling the case when a whole word in the input is masked,
and the same word in the output is split into multiple word pieces. 
The number of input and output elements would mismatch. 

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


## Compatibility

Tested on Ubuntu 16.04, Python 3.6, and torch==1.2.0
