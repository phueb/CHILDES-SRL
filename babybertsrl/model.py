
import numpy as np


class Model():

    def __init__(self, params, embeddings, num_labels):

        print('Initializing Allen NLP SRL model with in size = {} and out size = {}'.format(
            len(embeddings), num_labels))

        self.params = params
        vocab_size, embed_size = embeddings.shape

