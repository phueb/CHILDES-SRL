from collections import OrderedDict
import numpy as np
import pandas as pd
from babybertsrl import config


class Data:

    def __init__(self, params):
        self.params = params

        # ----------------------------------------------------------- words & labels

        self._word_set = set()  # holds words from both train and dev
        self._label_set = set()  # holds labels from both train and dev

        self.train_propositions = self.get_propositions_from_file(config.Data.train_data_path)
        self.dev_propositions = self.get_propositions_from_file(config.Data.dev_data_path)

        self.sorted_words = sorted(self._word_set)
        self.sorted_labels = sorted(self._label_set)

        self.sorted_words = [config.Data.pad_word, config.Data.unk_word] + self.sorted_words  # pad must have id=0
        self.sorted_labels = [config.Data.pad_label] + self.sorted_labels

        self.w2id = OrderedDict()  # word -> ID
        for n, w in enumerate(self.sorted_words):
            if w in self.w2id:
                raise SystemError('Trying to add word to w2id, but word is already in w2id')
            self.w2id[w] = n

        self.l2id = OrderedDict()  # label -> ID
        for n, l in enumerate(self.sorted_labels):
            if l in self.l2id:
                print('"{}" is already in l2id. Skipping'.format(l))
                continue  # the letter "O" should be assigned id=0 instead of last id
                # (this prevents overwriting existing entry with one pointing to the last id)
            self.l2id[l] = n
            if config.Data.verbose:
                print('"{:<12}" -> {:<4}'.format(l, n))

        assert len(self.w2id) == self.num_words

        # -------------------------------------------------------- console

        print('/////////////////////////////')
        print('Found {:,} training propositions ...'.format(self.num_train_propositions))
        print('Found {:,} dev propositions ...'.format(self.num_dev_propositions))
        print("Extracted {:,} train+dev words and {:,} labels".format(self.num_words, self.num_labels))

        for name, propositions in zip(['train', 'dev'],
                                      [self.train_propositions, self.dev_propositions]):
            lengths = [len(p[0]) for p in propositions]
            print("Max {} sentence length: {}".format(name, np.max(lengths)))
            print("Mean {} sentence length: {}".format(name, np.mean(lengths)))
            print("Median {} sentence length: {}".format(name, np.median(lengths)))
        print('/////////////////////////////')

        # -------------------------------------------------------- embeddings

        self.embeddings = self.make_embeddings()

        # -------------------------------------------------------- prepare data structures for training

        self.train = self.to_ids(self.train_propositions)
        self.dev = self.to_ids(self.dev_propositions)

    @property
    def num_labels(self):
        return len(self.sorted_labels)

    @property
    def num_words(self):
        return len(self.sorted_words)

    @property
    def num_train_propositions(self):
        return len(self.train_propositions)

    @property
    def num_dev_propositions(self):
        return len(self.dev_propositions)

    def get_propositions_from_file(self, file_path):
        """
        Read tokenized propositions from file.
          File format: {predicate_id} [word0, word1 ...] ||| [label0, label1 ...]
          Return:
            A list with elements of structure [[words], predicate, [labels]]
        """
        propositions = []
        with file_path.open('r') as f:

            for line in f.readlines():

                inputs = line.strip().split('|||')
                left_input = inputs[0].strip().split()
                right_input = inputs[1].strip().split()

                if config.Data.lowercase:
                    left_input = [w.lower() for w in left_input]

                if not config.Data.bio_tags:
                    right_input = [l.lstrip('-B').lstrip('-I') for l in right_input]

                # predicate
                predicate = int(left_input[0])

                # words + labels
                words = left_input[1:]
                labels = right_input

                if len(words) > self.params.max_sentence_length:
                    continue

                self._word_set.update(words)
                self._label_set.update(labels)

                propositions.append((words, predicate, labels))

        return propositions

    # ---------------------------------------------------------- embeddings

    def make_embeddings(self):

        assert len(self._word_set) > 0

        glove_p = config.RemoteDirs.root / (config.Data.glove_path_local or config.Data.glove_path)
        print('Loading word embeddings at:')
        print(glove_p)

        df = pd.read_csv(glove_p, sep=" ", quoting=3, header=None, index_col=0)
        w2embed = {key: val.values for key, val in df.T.items()}

        embedding_size = next(iter(w2embed.items()))[1].shape[0]
        print('Glove embedding size={}'.format(embedding_size))
        print('Num embeddings in GloVe file: {}'.format(len(w2embed)))

        # get embeddings for words in vocabulary
        res = np.zeros((self.num_words, embedding_size), dtype=np.float32)
        num_found = 0
        for w, row_id in self.w2id.items():
            try:
                word_embedding = w2embed[w]
            except KeyError:
                res[row_id] = np.random.standard_normal(embedding_size)
            else:
                res[row_id] = word_embedding
                num_found += 1

        print('Found {}/{} GloVe embeddings'.format(num_found, self.num_words))
        # if this number is extremely low, then it is likely that Glove txt file was only
        # partially copied to shared drive (copying should be performed in CL, not via nautilus)

        return res

    # --------------------------------------------------------- data structures for training a model

    def make_predicate_ids(self, proposition):
        """

        :param proposition: a tuple with structure (words, predicate, labels)
        :return: one-hot list, [sentence length]
        """
        offset = int(0)  # use + 1 if using sentence-beginning marker
        num_w_in_proposition = len(proposition[0])
        res = [int(i == proposition[1] + offset) for i in range(num_w_in_proposition)]
        return res

    def to_ids(self, propositions):
        """

        :param propositions: a tuple with structure (words, predicate, labels)
        :return: 3 lists, each of the same length, containing lists of integers
        """

        word_ids = []
        predicate_ids = []
        label_ids = []
        for proposition in propositions:
            word_ids.append([self.w2id[w] for w in proposition[0]])
            predicate_ids.append(self.make_predicate_ids(proposition))
            label_ids.append([self.l2id[l] for l in proposition[2]])

        return word_ids, predicate_ids, label_ids