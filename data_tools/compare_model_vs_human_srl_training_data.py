import re
import itertools
import pyprind

from babybertsrl import config

HUMAN_NAME = 'human-based-2008'
MODEL_NAME = 'childes-20191206'


# load human-based annotations
srl_path = config.Dirs.data / 'training' / f'{HUMAN_NAME}_srl.txt'
text = srl_path.read_text()
lines_h = text.split('\n')

# load model-based annotations
srl_path = config.Dirs.data / 'training' / f'{MODEL_NAME}_srl.txt'
text = srl_path.read_text()
lines_m = text.split('\n')[:-1]


def parse_line(l: str):
    """extract predicate location, words, and labels from a line in a training data file"""
    search = re.search('(\d) (.+) \|\|\| (.+)', l)
    if search is None:
        raise RuntimeError('Could not parse line')
    v_ind = search.group(1).split()
    words = search.group(2).split()
    tags_ = search.group(3).split()
    return v_ind, words, tags_


set_lines_h = set(lines_h)
set_lines_m = set(lines_m)
print(f'num human: {len(set_lines_h)}')
print(f'num model: {len(set_lines_m)}')
num_product = len(set_lines_h) * len(set_lines_m)
print(f'num product: {num_product}')

num_shared = 0
num_identical = 0
pbar = pyprind.ProgBar(num_product, stream=1, width=90, update_interval=10)
for line_h, line_m in itertools.product(set_lines_h, set_lines_m):

    tmp_m = line_m.split('|||')
    tmp_h = line_h.split('|||')
    vi_m, ws_m, ls_m = tmp_m[0][0], tmp_m[0][1:], tmp_m[1]
    vi_h, ws_h, ls_h = tmp_h[0][0], tmp_h[0][1:], tmp_h[1]

    if ws_m == ws_h and vi_m == vi_h:
        num_shared += 1

        print(''.join([f'{l:<12}' for l in ws_m.split()]))
        print(''.join([f'{l:<12}' for l in ls_m.split()]))
        print(''.join([f'{l:<12}' for l in ls_h.split()]))
        print()

        if ls_m == ls_h:
            num_identical += 1

    # pbar.update()


prop = num_identical / num_shared
print(f'Proportion identical={prop:.2f}')