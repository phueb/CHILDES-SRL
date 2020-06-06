from pathlib import Path

from babybertsrl.io import load_propositions_from_file, load_utterances_from_file


# ========================================================== SRL

root = Path(__file__).parent.parent
data_path_train_srl = root / 'data' / 'training' / f'childes-20191206_no-dev_srl.txt'
propositions = load_propositions_from_file(data_path_train_srl)

nouns_singular = set((root / 'analysis' / 'nouns_singular_annotator2.txt').open().read().split())
nouns_plural = set((root / 'analysis' / 'nouns_plural_annotator2.txt').open().read().split())

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

data_path_mlm = root / 'data' / 'training' / f'childes-20191206_mlm.txt'
utterances = load_utterances_from_file(data_path_mlm)

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