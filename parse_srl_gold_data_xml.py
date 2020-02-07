from pathlib import Path
import xml.etree.ElementTree as ET
import re
from nltk import Tree
from collections import deque
from typing import List, Any

from babybertsrl.srl_utils import make_srl_string

OUTSIDE_LABEL = '0'
XML_PATH = Path('data/babySRL-XML')
VERBOSE = False


def has_props(child):
    try:
        next(child.iterfind('{http://www.talkbank.org/ns/talkbank}props'))
    except StopIteration:
        return False
    else:
        return True


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
num_bad_head_loc = 0
num_bad_arg_loc = 0
num_total_good = 0
xy = []
for file_path in sorted(XML_PATH.glob('adam*.xml')):
    parse_tree = ET.parse(str(file_path))
    root = parse_tree.getroot()
    num_good_props_in_file = 0

    for utterance in root:
        if not has_props(utterance):
            continue

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
                    print('WARNING: Bad head location')  # TODO is the data really bad?
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
                        print('WARNING: Bad index. Skipping')  # TODO are there really bad data?
                        num_bad_arg_loc += 1
                        is_bad = True
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
            assert len(labels) == len(words)

            # console
            if VERBOSE:
                print(make_srl_string(words, labels))

            # collect
            num_good_props_in_file += 1
            xy.append((words, labels))

    print('Collected {} good propositions in {}'.format(num_good_props_in_file, file_path.name))
    num_total_good += num_good_props_in_file

print(f'num good              ={num_total_good:,}')
print(f'num no predicate      ={num_no_predicate:,}')
print(f'num bad head location ={num_bad_head_loc:,}')
print(f'num bad arg location  ={num_bad_arg_loc:,}')

# TODO align words and labels in a pandas DataFrame and keep in memory to feed directly to model
