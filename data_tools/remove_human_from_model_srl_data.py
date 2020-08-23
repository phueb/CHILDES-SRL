
from childes_srl import configs

HUMAN_NAME_1 = 'human-based-2008'
HUMAN_NAME_2 = 'human-based-2008'
MODEL_NAME = 'childes-20191206'


# load human-based annotations
srl_path1 = configs.Dirs.data / 'pre_processed' / f'{HUMAN_NAME_1}_srl.txt'
text1 = srl_path1.read_text()
propositions_h1 = [line.split('|||')[0] for line in text1.split('\n')]

# load human-based annotations
srl_path2 = configs.Dirs.data / 'pre_processed' / f'{HUMAN_NAME_2}_srl.txt'
text2 = srl_path2.read_text()
propositions_h2 = [line.split('|||')[0] for line in text2.split('\n')]

# load model-based annotations
srl_path3 = configs.Dirs.data / 'pre_processed' / f'{MODEL_NAME}_srl.txt'
text3 = srl_path3.read_text()
lines_m = text3.split('\n')[:-1]
propositions_m = [line.split('|||')[0] for line in lines_m]


# make sets
propositions_h = propositions_h1 + propositions_h2
set_propositions_h = set(propositions_h)
set_propositions_m = set(propositions_m)
print(f'num human unique propositions: {len(set_propositions_h):>9,}/{len(propositions_h):>9,}')
print(f'num model unique propositions: {len(set_propositions_m):>9,}/{len(propositions_m):>9,}')

# exclude shared
lines = []
num_excluded = 0
for line_m in set(lines_m):
    prop_m = line_m.split('|||')[0]
    if prop_m not in set_propositions_h:
        lines.append(line_m)
    else:
        num_excluded += 1

print(f'Excluded {num_excluded:>9,}/{len(lines_m):>9,}')

# write non-shared to file
print(f'Writing {len(lines)} lines to file...')
srl_path = configs.Dirs.data / 'pre_processed' / f'{MODEL_NAME}_no-dev_srl.txt'
with srl_path.open('w') as f:
    for n, line in enumerate(lines):
        f.write(line)
        if n + 1 != len(lines):  # do not write '\n' at end of file
            f.write('\n')