from typing import List
import pandas as pd
import re

from babybertsrl import config

CORPUS_NAME = 'childes-20191206'


def make_srl_string(words: List[str],
                    tags: List[str]) -> str:
    frame = []
    chunk = []

    for (token, tag) in zip(words, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                frame.append("[" + " ".join(chunk) + "]")
                chunk = []

            if tag.startswith("B-"):
                chunk.append(tag[2:] + ": " + token)
            elif tag == "O":
                frame.append(token)

    if chunk:
        frame.append("[" + " ".join(chunk) + "]")

    return " ".join(frame)


# load annotations
srl_path = config.Dirs.data / 'CHILDES' / f'{CORPUS_NAME}_srl.txt'
text = srl_path.read_text()

# load previously checked annotations
path = config.Dirs.root / 'srl_check.csv'
df = pd.read_csv(path, index_col=False)

print(df)

for line in text.split('\n'):

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

print('Saving the following df:')
print(df)
df.to_csv(path, index=False)
print('Done')