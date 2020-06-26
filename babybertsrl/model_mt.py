from typing import Dict, List, Any
import torch
import numpy as np
from torch.nn import Linear, Dropout, functional as F
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertOnlyMLMHead

from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients

from babybertsrl.word_pieces import convert_wordpieces_to_words
from babybertsrl import configs


class MTBert(torch.nn.Module):
    """
    Multi-task BERT.
    It has a head for MLM and another head for SRL, and can be trained jointly on both tasks
    """

    def __init__(self,
                 effective_vocab_mlm: Vocabulary,
                 effective_vocab_srl: Vocabulary,
                 bert_model: BertModel,
                 embedding_dropout: float = 0.0,
                 ) -> None:

        super().__init__()
        self.bert_model = bert_model

        self.effective_vocab_mlm = effective_vocab_mlm
        self.effective_vocab_srl = effective_vocab_srl

        # labels namespace is 2 elements shorter than tokens because it does not have @@PADDING@@ and @@UNKNOWN@@
        self.num_out_srl = effective_vocab_srl.get_vocab_size('labels')

        # make one projection layer for each task
        # self.projection_layer_mlm = Linear(self.bert_model.config.hidden_size, self.num_out_mlm)
        self.projection_layer_srl = Linear(self.bert_model.config.hidden_size, self.num_out_srl)
        self.mlm_head = BertOnlyMLMHead(self.bert_model.config, self.bert_model.embeddings.word_embeddings.weight)

        self.embedding_dropout = Dropout(p=embedding_dropout)

    def forward(self,
                task: str,
                tokens: Dict[str, torch.Tensor],
                indicator: torch.Tensor,  # indicates either masked word, or predicate
                metadata: List[Dict[str, Any]],
                compute_probabilities: bool = False,
                tags: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        task: string indicating which projection layer to use: either "srl" or "mlm"
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the masked token or predicate
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no mask.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata contains the original words in the sentence, the masked word or predicate,
             and start offsets for converting wordpieces back to a sequence of words.
        compute_probabilities: whether or not to compute probabilities using softmax
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        # move to GPU
        tokens['tokens'] = tokens['tokens'].cuda()
        indicator = indicator.cuda()
        if tags is not None:
            tags = tags.cuda()

        # get BERT contextualized embeddings
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(input_ids=tokens['tokens'],
                                             token_type_ids=indicator,
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()

        # use correct head for task
        if task == 'mlm':
            # logits = self.projection_layer_mlm(embedded_text_input)
            logits = self.mlm_head(bert_embeddings)  # projects to vector of size bert_config.vocab_size

            if tags is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(logits.view(-1, self.bert_model.config.vocab_size), tags.view(-1))

            reshaped_logits = logits.view(-1, self.bert_model.config.vocab_size)  # collapse time steps and batches
            class_probabilities = F.softmax(reshaped_logits, dim=-1).view([batch_size,
                                                                           sequence_length,
                                                                           self.bert_model.config.vocab_size])

        elif task == 'srl':
            logits = self.projection_layer_srl(embedded_text_input)

            if tags is not None:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)

            # TODO only compute softmax when probing (not when training)
            reshaped_logits = logits.view(-1, self.num_out_srl)  # collapse time steps and batches
            class_probabilities = F.softmax(reshaped_logits, dim=-1).view([batch_size,
                                                                           sequence_length,
                                                                           self.num_out_srl])
        else:
            raise AttributeError('Invalid arg to "task"')

        output_dict = {"logits": logits,
                       "mask": mask,         # for decoding
                       'start_offsets': [],  # for decoding BIO SRL tags
                       'in': [],
                       'gold_tags': [],     # for testing
                       'gold_tags_wp': [],  # for testing
                       }

        # add meta data to output
        for d in metadata:
            output_dict['in'].append(d['in'])
            output_dict['gold_tags'].append(d['gold_tags'])
            output_dict['gold_tags_wp'].append(d['gold_tags_wp'])
            output_dict['start_offsets'].append(d['start_offsets'])

        if tags is not None:
            output_dict['loss'] = loss

        if compute_probabilities:
            output_dict["class_probabilities"] = class_probabilities  # defined over word-pieces

        return output_dict

    def decode_mlm(self,
                   output_dict: Dict[str, Any],
                   ) -> List[List[str]]:
        """
        No viterbi decoding when task is MLM
        """

        # get probabilities
        class_probabilities = output_dict['class_probabilities']
        class_probabilities_cpu = [class_probabilities[i].detach().cpu().numpy()
                                   for i in range(class_probabilities.size(0))]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()

        # decode = convert back from wordpieces
        res = []
        for predictions, length, offsets, gold_tags, gold_tags_wp in zip(class_probabilities_cpu,
                                                                         sequence_lengths,
                                                                         output_dict['start_offsets'],
                                                                         output_dict['gold_tags'],
                                                                         output_dict['gold_tags_wp']):
            tag_wp_probabilities = predictions[:length]
            tag_wp_ids = np.argmax(tag_wp_probabilities, axis=1)
            tags_wp = [self.effective_vocab_mlm.get_token_from_index(tag_wp_id, namespace="labels")
                       for tag_wp_id in tag_wp_ids]
            # note: softmax is over wordpiece tokenizer vocab, but tokens are retrieved from Allen NLP vocab.
            # this works because Allen NLP vocab uses indices obtained from word-piece tokenizer

            if configs.Wordpieces.verbose:
                print('Converting wordpieces back to words:')
                print('gold_tags')
                print(gold_tags)
                print('gold_tags_wp')
                print(gold_tags_wp)

            predicted_tags = convert_wordpieces_to_words(tags_wp)
            if configs.Wordpieces.warn_on_mismatch:
                if len(gold_tags) != len(predicted_tags):
                    raise RuntimeError('Number of whole words in decoded output does not match number in input.')
            res.append(predicted_tags)

        return res  # list of predicted whole words, one per sequence in batch

    def decode_srl(self,
                   output_dict: Dict[str, Any],
                   ) -> List[List[str]]:
        """
        Do NOT use decoding constraints - transition matrix has zeros only
        we are interested in learning dynamics, not best performance.
        Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
        """

        # get probabilities
        class_probabilities = output_dict['class_probabilities']
        class_probabilities_cpu = [class_probabilities[i].detach().cpu() for i in range(class_probabilities.size(0))]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()

        # effective_vocab
        effective_vocab = self.effective_vocab_srl
        num_out = self.num_out_srl

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        all_labels = effective_vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        assert num_out == num_labels
        transition_matrix = torch.zeros([num_labels, num_labels])

        # decode = get max likelihood tags + convert back from wordpieces
        res = []
        for predictions, length, offsets, gold_tags in zip(class_probabilities_cpu,
                                                           sequence_lengths,
                                                           output_dict['start_offsets'],
                                                           output_dict['gold_tags']):
            # get max likelihood tags
            tag_wp_probabilities = predictions[:length]
            ml_tag_wp_ids, _ = viterbi_decode(tag_wp_probabilities, transition_matrix)  # ml = max likelihood
            ml_tags_wp = [effective_vocab.get_token_from_index(tag_id, namespace="labels") for tag_id in ml_tag_wp_ids]
            # convert back from wordpieces
            ml_tags = [ml_tags_wp[i] for i in offsets]  # specific to BIO SRL tags
            res.append(ml_tags)

        return res  # list of max likelihood tags

    def train_on_batch(self, task, batch, optimizer):
        # forward + loss
        optimizer.zero_grad()
        output_dict = self(task, **batch)  # input is dict[str, tensor]
        loss = output_dict['loss']
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        # backward + update
        loss.backward()
        rescale_gradients(self, grad_norm=1.0)
        optimizer.step()

        return loss