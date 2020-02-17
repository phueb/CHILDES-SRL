from typing import Dict, List, Any
import torch
from torch.nn import Linear, Dropout, functional as F
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients


class MTBert(torch.nn.Module):
    """
    Multi-task BERT.
    It is trained jointly on SRL and MLM tasks, without pre-training.
    """

    def __init__(self,
                 vocab_mlm: Vocabulary,
                 vocab_srl: Vocabulary,
                 bert_model: BertModel,
                 embedding_dropout: float = 0.0,
                 ) -> None:

        super().__init__()
        self.bert_model = bert_model

        self.vocab_mlm = vocab_mlm
        self.vocab_srl = vocab_srl

        # labels namespace is 2 elements shorter than tokens because it does not have PADDING and UNKNOWN
        self.num_out_mlm = vocab_mlm.get_vocab_size('labels')
        self.num_out_srl = vocab_srl.get_vocab_size('labels')

        # make one projection layer for each task
        self.projection_layer_mlm = Linear(self.bert_model.config.hidden_size, self.num_out_mlm)
        self.projection_layer_srl = Linear(self.bert_model.config.hidden_size, self.num_out_srl)

        self.embedding_dropout = Dropout(p=embedding_dropout)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                indicator: torch.Tensor,  # indicates either masked word, or predicate
                metadata: List[Dict[str, Any]],
                task: str,
                tags: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
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
        task: string indicating which projection layer to use: either "srl" or "mlm"
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
            logits = self.projection_layer_mlm(embedded_text_input)
            num_out = self.num_out_mlm
        elif task == 'srl':
            logits = self.projection_layer_srl(embedded_text_input)
            num_out = self.num_out_srl
        else:
            raise AttributeError('Invalid arg to "task"')

        # compute output
        reshaped_logits = logits.view(-1, num_out)  # collapse time steps and batches
        class_probabilities = F.softmax(reshaped_logits, dim=-1).view([batch_size,
                                                                       sequence_length,
                                                                       num_out])

        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,  # defined over word-pieces
                       "mask": mask,         # for decoding
                       'start_offsets': [],  # for decoding
                       'in': [],
                       'gold_tags': [],
                       }

        # add meta data to output
        for d in metadata:
            output_dict['in'].append(d['in'])
            output_dict['gold_tags'].append(d['gold_tags'])
            output_dict['start_offsets'].append(d['start_offsets'])

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask)
            output_dict['loss'] = loss
        return output_dict

    def decode(self,
               output_dict: Dict[str, Any],
               task: str,
               ) -> List[List[str]]:
        """
        Do NOT use decoding constraints - transition matrix has zeros only
        we are interested in learning dynamics, not best performance.
        Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
        """

        # get probabilities
        all_predictions = output_dict['class_probabilities']
        predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()

        # vocab
        if task == 'mlm':
            vocab = self.vocab_mlm
            num_out = self.num_out_mlm
        elif task == 'srl':
            vocab = self.vocab_srl
            num_out = self.num_out_srl
        else:
            raise AttributeError('Invalid arg to "task"')

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        all_labels = vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        assert num_out == num_labels
        transition_matrix = torch.zeros([num_labels, num_labels])

        # decode
        tags = []
        for predictions, length, offsets, gold_tags in zip(predictions_list,
                                                           sequence_lengths,
                                                           output_dict['start_offsets'],
                                                           output_dict['gold_tags']):
            tag_probabilities = predictions[:length]
            max_likelihood_tag_ids, _ = viterbi_decode(tag_probabilities,
                                                       transition_matrix)
            tags_word_pieces = [vocab.get_token_from_index(x, namespace="labels")
                                for x in max_likelihood_tag_ids]
            tags.append([tags_word_pieces[i] for i in offsets])

        return tags

    def train_on_batch(self, batch, optimizer):
        # forward + loss
        optimizer.zero_grad()
        output_dict = self(**batch)  # input is dict[str, tensor]
        loss = output_dict['loss']
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        # backward + update
        loss.backward()
        rescale_gradients(self, grad_norm=1.0)
        optimizer.step()

        return loss