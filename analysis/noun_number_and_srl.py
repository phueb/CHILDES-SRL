"""
What proportion of nouns are singular and plural?
Results are organized by semantic role.
"""
from pathlib import Path

from childes_srl.io import load_srl_data, load_mlm_data

root = Path(__file__).parent.parent

nouns_singular = set((root / 'data' / 'word_lists' / 'nouns_singular.txt').open().read().split())
nouns_plural = set((root / 'data' / 'word_lists' / 'nouns_plural.txt').open().read().split())

# ========================================================== SRL

data_path_train_srl = root / 'data' / 'pre_processed' / f'childes-20191206_no-dev_srl.txt'
propositions = load_srl_data(data_path_train_srl)

tag2number = {}
for words, pred_id, tags in propositions:
    for w, t in zip(words, tags):
        if w in nouns_singular:
            tag2number.setdefault(t, {'s': 0, 'p': 0})['s'] += 1
        elif w in nouns_plural:
            tag2number.setdefault(t, {'s': 0, 'p': 0})['p'] += 1

for tag, number in tag2number.items():
    total = number['s'] + number['p']
    print(f'{tag:<16} s={number["s"]/total:.2f} p={number["p"]/total:.2f}')

# ========================================================== MLM

data_path_mlm = root / 'data' / 'raw' / 'childes' / f'childes-20191206.txt'
utterances = load_mlm_data(data_path_mlm)

num_s = 0
num_p = 0
for u in utterances:
    for w in u:
        if w in nouns_singular:
            num_s += 1
        elif w in nouns_plural:
            num_p += 1

total = num_p + num_s
print(f's={num_s/total:.2f} p={num_p/total:.2f}')