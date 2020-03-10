from functools import reduce
from operator import iconcat
from collections import Counter

from babybertsrl import config

MODEL_NAME = 'childes-20191206'

# load model-based annotations
srl_path = config.Dirs.data / 'training' / f'{MODEL_NAME}_no-dev_srl.txt'
text = srl_path.read_text()
lines = text.split('\n')[:-1]

# get a list of all tags
tag_lines = [line.split('|||')[1].split() for line in lines]
tags = reduce(iconcat, tag_lines, [])  # flatten list of lists
print(f'num tags={len(tags):>9,}')

# remove "B-" and "I-"
tags_no_bio = [tag.lstrip('B-').lstrip('I-') for tag in tags]

# count
t2f = Counter(tags_no_bio)
for t, f in sorted(t2f.items(), key=lambda i: i[1]):
    print(f'{t:<12} occurs {f:>9,} times')




