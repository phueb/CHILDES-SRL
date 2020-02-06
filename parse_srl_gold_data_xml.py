from pathlib import Path
import xml.etree.ElementTree as ET
import re
from nltk import Tree

from babybertsrl.srl_utils import make_srl_string

OUTSIDE_LABEL = '0'
XML_PATH = Path('data/babySRL-XML')


def has_props(child):
    try:
        next(child.iterfind('{http://www.talkbank.org/ns/talkbank}props'))
    except StopIteration:
        return False
    else:
        return True


xy = []

for file_path in sorted(XML_PATH.glob('adam*.xml')):
    parse_tree = ET.parse(str(file_path))
    root = parse_tree.getroot()
    num_props = 0

    for utterance in root:
        if not has_props(utterance):
            continue

        # collect words in utterance
        words = []
        for word in utterance.iter('{http://www.talkbank.org/ns/talkbank}w'):
            words += word.text.split("'")

            # TODO also include commas  e.g. <pause symbolic-length="simple"/>

        print(words)

        # get parse tree
        parse_string = utterance.find('{http://www.talkbank.org/ns/talkbank}parse').text
        parse_tree = Tree.fromstring(parse_string)

        # collect label sequence for each <proposition> in the utterance
        sense2labels = {}
        for proposition in utterance.iter('{http://www.talkbank.org/ns/talkbank}proposition'):

            # initialize label-sequence
            sense = proposition.attrib['sense']
            label_text_list = list(proposition.itertext())
            labels = [OUTSIDE_LABEL for _ in range(len(label_text_list))]
            sense2labels[sense] = labels

            # loop over arguments in the proposition - reconstructing label-sequence along the way
            for label_text in label_text_list:

                # parse label_text
                res = re.findall(r'(\d+):(\d)-(.*)', label_text)[0]
                loc = int(res[0])  # location in sentence of first word in span
                num_up = int(res[1])  # levels up in hierarchy at which all sister-trees are part of span
                label = str(res[2])
                print(loc, num_up, label)

                if label == 'rel':
                    labels[loc] = label
                else:

                    tp = parse_tree.leaf_treeposition(loc)
                    argument_tree = parse_tree[tp[:-num_up]]
                    print(argument_tree)
                    labels[loc:loc + len(argument_tree.leaves())] = [label]

            print(labels)

            # checks
            assert len(labels) == len(words)
            assert labels.count('rel') == 1

            # console
            print(make_srl_string(words, labels))
            print()

            # collect
            num_props += 1
            xy.append((words, labels))

    print(xy[:10])
    print('Found {} propositions in {}'.format(num_props, file_path.name))

    raise SystemExit

# TODO align words and labels in a pandas DataFrame and keep in memory to feed directly to model
