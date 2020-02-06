import random
import pandas as pd
import re

from babybertsrl.srl_utils import make_srl_string
from babybertsrl import config

CORPUS_NAME = 'childes-20191206'


# load annotations
srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_srl.txt'
text = srl_path.read_text()
lines = text.split('\n')

# load previously checked annotations
path = config.Dirs.root / 'srl_check.csv'
df = pd.read_csv(path, index_col=False)

print(f'Checked {len(df):,}/{len(lines):,} lines')

for line in random.sample(lines, k=len(lines)):

    if df['line'].isin([line]).any():
        print('Checked this line before. Skipping')
        continue

    search = re.search('(\d) (.+) \|\|\| (.+)', line)
    if not search:
        print('WARNING: Regex could not parse line')
        continue

    v_ind = search.group(1).split()
    words = search.group(2).split()
    tags_ = search.group(3).split()

    print(make_srl_string(words, tags_))
    key = input('...')
    if key == 'q':
        break
    elif key == 'b':  # bad
        is_bad = True
    else:
        is_bad = False

    row = pd.DataFrame(data={'line': [line], 'is_bad': [is_bad]})
    df = df.append(row, ignore_index=True, sort=False)

    df.to_csv(path, index=False)
    print('Saved df')

num_good = df.is_bad.value_counts()[0]
num_bad = df.is_bad.value_counts()[1]
prop = num_good / len(df)
print(f'Proportion correct={prop:.2f}')