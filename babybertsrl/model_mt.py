from typing import Dict, List, Any
import torch
import numpy as np
from torch.nn import Linear, Dropout, functional as F
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertOnlyMLMHead

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
                 id2tag_wp_srl: Dict[int, str],
                 id2tag_wp_mlm: Dict[int, str],
                 bert_model: BertModel,
                 embedding_dropout: float = 0.0,
                 ) -> None:

        super().__init__()
        self.bert_model = bert_model

        # vocab for heads
        self.id2tag_wp_srl = id2tag_wp_srl
        self.id2tag_wp_mlm = id2tag_wp_mlm
        # Allen NLP vocab gives same word indices as word-piece tokenizer
        # because indices are obtained from word-piece tokenizer during conversion to instances

        # make one projection layer for each task
        self.head_srl = Linear(self.bert_model.config.hidden_size, len(self.id2tag_wp_srl))
        self.head_mlm = BertOnlyMLMHead(self.bert_model.config, self.bert_model.embeddings.word_embeddings.weight)

        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.xe = CrossEntropyLoss(ignore_index=configs.Training.ignored_index)  # ignore tags with index=ignore_index

    def forward(self,
                task: str,
                tokens: Dict[str, torch.Tensor],
                indicator: torch.Tensor,  # indicates either masked word, or predicate
                metadata: List[Dict[str, Any]],
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
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
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
            logits = self.head_mlm(bert_embeddings)  # projects to vector of size bert_config.vocab_size
            if tags is not None:
                loss = self.xe(logits.view(-1, self.bert_model.config.vocab_size), tags.view(-1))

        elif task == 'srl':
            logits = self.head_srl(embedded_text_input)
            if tags is not None:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)
        else:
            raise AttributeError('Invalid arg to "task"')

        output_dict = {"logits": logits,
                       "mask": mask,         # for decoding
                       'start_offsets': [],  # for decoding BIO SRL tags
                       'in': [],
                       'gold_tags': [],     # for debugging
                       'gold_tags_wp': [],  # for debugging
                       }

        # add meta data to output
        for d in metadata:
            output_dict['in'].append(d['in'])
            output_dict['gold_tags'].append(d['gold_tags'])
            output_dict['gold_tags_wp'].append(d['gold_tags_wp'])
            output_dict['start_offsets'].append(d['start_offsets'])

        if tags is not None:
            output_dict['loss'] = loss

        return output_dict

    def decode_mlm(self,
                   output_dict: Dict[str, Any],
                   ) -> List[List[str]]:
        """
        :returns original sequence with [MASK] replaced with highest scoring word-piece.
        No viterbi or handling word-piece sequences, because task is MLM, not SRL.
        """

        res = []
        for seq_id, mlm_in in enumerate(output_dict['in']):

            # get predicted wp
            masked_id = mlm_in.index('[MASK]')
            logits_in_sequence = output_dict['logits'][seq_id]
            logits_for_masked_wp = logits_in_sequence[masked_id].detach().cpu().numpy()  # shape is now [vocab_size]
            tag_wp_id = np.asscalar(np.argmax(logits_for_masked_wp))
            tag_wp = self.id2tag_wp_mlm[tag_wp_id]

            # fill in input sequence
            filled_in_sequence = mlm_in.copy()
            filled_in_sequence[masked_id] = tag_wp
            res.append(filled_in_sequence)

        return res  # sequence with predicted word-piece, one per sequence in batch

    def decode_srl(self,
                   output_dict: Dict[str, Any],
                   ) -> List[List[str]]:
        """
        for each sequence in batch:
        1) get max likelihood tags
        2) convert back from wordpieces


        Do NOT use decoding constraints - transition matrix has zeros only
        we are interested in learning dynamics, not best performance.
        Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
        """

        # get probabilities
        logits = output_dict['logits']
        reshaped_logits = logits.view(-1, len(self.id2tag_wp_srl))  # collapse time steps and batches
        class_probabilities = F.softmax(reshaped_logits, dim=-1).view([logits.shape[0],
                                                                       logits.shape[1],
                                                                       len(self.id2tag_wp_srl)])
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        transition_matrix = torch.zeros([len(self.id2tag_wp_srl), len(self.id2tag_wp_srl)])

        # loop over each sequence in batch
        res = []
        for seq_id in range(logits.shape[0]):
            # get max likelihood tags
            length = sequence_lengths[seq_id]
            tag_wp_probabilities = class_probabilities[seq_id].detach().cpu()[:length]

            # TODO debug
            print(class_probabilities.shape)
            print(tag_wp_probabilities.shape)

            ml_tag_wp_ids, _ = viterbi_decode(tag_wp_probabilities, transition_matrix)  # ml = max likelihood
            ml_tags_wp = [self.id2tag_wp_srl[tag_id] for tag_id in ml_tag_wp_ids]
            # convert back from wordpieces
            ml_tags = [ml_tags_wp[i] for i in  output_dict['start_offsets'][seq_id]]  # specific to BIO SRL tags
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