from pathlib import Path
import xml.etree.ElementTree as ET
import re
from nltk import Tree
from collections import deque
from typing import List, Any

from babybertsrl.srl_utils import make_srl_string
from babybertsrl import config

NAME = 'human-based-2018'
XML_PATH = Path(f'data/srl_{NAME}/xml')
VERBOSE = False
EXCLUDE_CHILD = True
OUTSIDE_LABEL = '0'


def has_props(e):
    try:
        next(e.iterfind('{http://www.talkbank.org/ns/talkbank}props'))
    except StopIteration:
        return False
    else:
        return True


def is_child(e):
    """is utterance spoken by child?"""
    if e.attrib['who'] == 'CHI':
        return True
    else:
        return False


def get_start_index(a: List[Any],
                    b: List[Any],
                    ) -> int:
    """return index into "a" which is first location of section of "a" which matches "b". """
    num_b = len(b)
    init_length = num_b - 1
    d = deque(a[:init_length], maxlen=num_b)
    for n, ai in enumerate(a[init_length:]):
        d.append(ai)
        if list(d) == b:
            return n
    else:
        raise ValueError('a does not contain b')


num_no_predicate = 0
num_no_arguments = 0
num_bad_head_loc = 0
num_bad_arg_loc = 0
num_prepositions = 0
num_total_good = 0
lines = []
for file_path in sorted(XML_PATH.rglob('*.xml')):
    parse_tree = ET.parse(str(file_path))
    root = parse_tree.getroot()
    num_good_props_in_file = 0

    for utterance in root:
        if not has_props(utterance):
            print('WARNING: Did not find propositions. Skipping')
            continue
        if is_child(utterance) and EXCLUDE_CHILD:
            print('WARNING: Skipping child utterance')

        # get parse tree
        parse_string = utterance.find('{http://www.talkbank.org/ns/talkbank}parse').text
        parse_tree = Tree.fromstring(parse_string)

        # words - get them from parse tree because parsing xml is difficult
        words = parse_tree.leaves()

        if VERBOSE:
            print()
            print('=============================================')
            print(f'{file_path.name} {utterance.attrib["uID"]}')
            print(' '.join(words))
            print('=============================================')
            print()

        # collect label sequence for each <proposition> in the utterance
        sense2labels = {}
        for proposition in utterance.iter('{http://www.talkbank.org/ns/talkbank}proposition'):

            if proposition.attrib['lemma'].endswith('-p'):  # TODO what to do here?
                print('WARNING: Skipping prepositional proposition')
                num_prepositions += 1
                continue

            if VERBOSE:
                print(proposition.attrib)

            # initialize label-sequence
            sense = proposition.attrib['sense']
            label_text_list = list(proposition.itertext())
            labels = [OUTSIDE_LABEL for _ in range(len(words))]
            sense2labels[sense] = labels
            is_bad = False

            # loop over arguments in the proposition - reconstructing label-sequence along the way
            for label_text in label_text_list:

                # parse label_text
                res = re.findall(r'(\d+):(\d)-(.*)', label_text)[0]
                head_loc = int(res[0])  # location in sentence of head (not first word) of argument
                num_up = int(res[1])  # levels up in hierarchy at which all sister-trees are part of argument span
                tag = str(res[2])

                if VERBOSE:
                    print(f'{head_loc:>2} {num_up:>2} {tag:>12}')

                try:
                    words[head_loc]
                except IndexError:
                    print('WARNING: Bad head location')
                    num_bad_head_loc += 1
                    is_bad = True
                    break

                if 'rel' in tag:
                    labels[head_loc] = 'B-V'
                else:
                    tp = parse_tree.leaf_treeposition(head_loc)
                    argument_tree = parse_tree[tp[: - num_up - 1]]   # go up in tree from head of current argument
                    argument_length = len(argument_tree.leaves())
                    argument_labels = [f'B-{tag}'] + [f'I-{tag}'] * (argument_length - 1)
                    start_loc = get_start_index(words, argument_tree.leaves())

                    if not labels[start_loc: start_loc + argument_length] == [OUTSIDE_LABEL] * argument_length:
                        print('WARNING: Bad argument location. Skipping')
                        num_bad_arg_loc += 1
                        is_bad = True

                        # print(labels)
                        # labels[start_loc: start_loc + argument_length] = argument_labels
                        # print(labels)
                        #
                        # if input():
                        #     continue

                        break
                    labels[start_loc: start_loc + argument_length] = argument_labels

            if is_bad:
                continue

            # pre-check console
            if VERBOSE:
                for w, l in zip(words, labels):
                    print(f'{w:<12} {l:<12}')

            # checks
            if labels.count('B-V') != 1:
                print('WARNING: Did not find predicate')
                num_no_predicate += 1
                continue

            if sum([1 if l.startswith('B-ARG') else 0 for l in labels]) == 0:
                print('WARNING: Did not find arguments')
                num_no_arguments += 1
                continue

            assert len(labels) == len(words)

            # console
            if VERBOSE:
                print(make_srl_string(words, labels))

            # make line
            verb_index = labels.index('B-V')
            x_string = " ".join(words)
            y_string = " ".join(labels)
            line = f'{verb_index} {x_string} ||| {y_string}'

            # collect
            num_good_props_in_file += 1
            lines.append(line)

    print('Collected {} good propositions in {}'.format(num_good_props_in_file, file_path.name))
    num_total_good += num_good_props_in_file

print(f'num good              ={num_total_good:,}')
print(f'num no arguments      ={num_no_arguments:,}')
print(f'num no predicate      ={num_no_predicate:,}')
print(f'num bad head location ={num_bad_head_loc:,}')
print(f'num bad arg location  ={num_bad_arg_loc:,}')
print(f'num prepositions      ={num_prepositions:,}')


print(f'Writing {len(lines)} lines to file...')
srl_path = config.Dirs.data / 'training' / f'{NAME}_srl.txt'
with srl_path.open('w') as f:
    for line in lines:
        f.write(line + '\n')