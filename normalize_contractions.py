"""
The CHILDES text files have been generated using the spacy tokenizer.
The spacy default tokenizer splits on contractions, meaning that tokens like "'ll" are
whitespace-separated from tokens like "he".
in order to use an SRL tagger, non-English tokens must be converted into English words.
for example, "'ll" must be converted into "will".

extra care is taken with the token "'s" because it can either be "is" or "us", depending on context.
"""

from babybertsrl import config
import re

