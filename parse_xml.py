from pathlib import Path
import xml.etree.ElementTree as ET
import re

UNKNOWN_LABEL = '0'


adam_p = Path('data/BabySRL-XML/Adam')
eve_p = Path('data/BabySRL-XML/Eve')
sarah_p = Path('data/BabySRL-XML/Eve')


# TODO make thsi into a prepare_data function whcih is called by ludwigcluster before submission of jobs


def has_props(child):
    try:
        next(child.iterfind('{http://www.talkbank.org/ns/talkbank}props'))
    except StopIteration:
        return False
    else:
        return True


xy = []

for xml_p in sorted(adam_p.glob('*.xml')):
    tree = ET.parse(str(xml_p))
    root = tree.getroot()
    num_props = 0

    for c1 in root:
        if has_props(c1):

            # collect words
            words = []  # in a single utterance
            for c2 in c1:  # iterate over children in utterance node
                if c2.tag == '{http://www.talkbank.org/ns/talkbank}w':
                    words_ = c2.text.split("'")
                    print(words_)
                    words += words_

            # collect labels
            labels = [UNKNOWN_LABEL] * len(words)  # in a single utterance
            for c2 in c1:  # iterate over children in utterance node
                if c2.tag == '{http://www.talkbank.org/ns/talkbank}props':
                    for label_child in c2.itertext():
                        print(label_child)

                        res = re.findall(r'(\d+):(\d)-(.*)', label_child)[0]
                        start = int(res[0])
                        length = int(res[1]) + 1
                        label = str(res[2])

                        label_span = [label] * length

                        for i in range(start, min(start + length, len(labels))):
                            # noinspection PyTypeChecker
                            labels[i] = label

                    print(labels)
                    assert len(labels) == len(words)

                    assert len([label for label in labels if label == 'rel']) == 1
                    # TODO handle multiple propositions (multiple "rel" labels)

            print()
            num_props += 1
            xy.append((words, labels))

    print(xy[:10])
    print('Found {} propositions in {}'.format(num_props, xml_p.name))

    raise SystemExit

# TODO align words and labels in a pandas DataFrame and keep in memory to feed directly to model
