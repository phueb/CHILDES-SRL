from pathlib import Path
import xml.etree.ElementTree as ET
import re

UNKNOWN_LABEL = '0'


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
    tree = ET.parse(str(file_path))
    root = tree.getroot()
    num_props = 0

    for c1 in root:
        if has_props(c1):

            # collect words
            words = []  # in a single utterance
            for c2 in c1:  # iterate over children in utterance node
                if c2.tag == '{http://www.talkbank.org/ns/talkbank}w':
                    words_ = c2.text.split("'")
                    # print(words_)
                    words += words_

            print(words)

            # collect labels
            labels = [UNKNOWN_LABEL] * len(words)  # in a single utterance
            for c2 in c1:  # iterate over children in utterance node
                if c2.tag == '{http://www.talkbank.org/ns/talkbank}props':
                    for label_child in c2.itertext():
                        print(label_child)

                        res = re.findall(r'(\d+):(\d)-(.*)', label_child)[0]
                        loc = int(res[0])
                        num_up = int(res[1])  # levels up in hierarchy at which all sister-trees are art of span
                        label = str(res[2])


                    print(labels)
                    assert len(labels) == len(words)

                    assert len([label for label in labels if label == 'rel']) == 1
                    # TODO handle multiple propositions (multiple "rel" labels)

            print()
            num_props += 1
            xy.append((words, labels))

    print(xy[:10])
    print('Found {} propositions in {}'.format(num_props, file_path.name))

    raise SystemExit

# TODO align words and labels in a pandas DataFrame and keep in memory to feed directly to model
