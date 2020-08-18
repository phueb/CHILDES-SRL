from typing import Dict, List, Any
import torch
import numpy as np
from torch.nn import Linear, Dropout, functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertModel, BertOnlyMLMHead

from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients

from babybertsrl import configs


class MTBert(torch.nn.Module):
    """
    Multi-task BERT.
    It has a head for MLM and another head for SRL, and can be trained jointly on both tasks
    """

    def __init__(self,
                 id2tag_wp_mlm: Dict[int, str],
                 id2tag_wp_srl: Dict[int, str],
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
        self.head_mlm = BertOnlyMLMHead(self.bert_model.config)

        self.embedding_dropout = Dropout(p=embedding_dropout)

        self.xe = CrossEntropyLoss(ignore_index=configs.Training.ignored_index)  # ignore tags with index=ignore_index
        self.forced_choice_xe = CrossEntropyLoss(ignore_index=configs.Training.ignored_index-1,   # nothing is ignored
                                                 reduction='none')  # keep per-utterance xe scores

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
            all zeros, in the case that the sentence has no mask.  # TODO so is this required even for MLM?
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

        loss = None

        # move to GPU
        tokens['tokens'] = tokens['tokens'].cuda()
        indicator = indicator.cuda()
        if tags is not None:
            tags = tags.cuda()

        # get BERT contextualized embeddings
        attention_mask = get_text_field_mask(tokens)
        outputs = self.bert_model(input_ids=tokens['tokens'],
                                  token_type_ids=indicator,
                                  attention_mask=attention_mask
                                  )
        bert_embeddings = outputs[0]
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()

        # for MLM training
        if task == 'mlm':
            print(bert_embeddings.shape)
            logits = self.head_mlm(bert_embeddings)  # projects to vector of size bert_config.vocab_size
            if tags is not None:
                loss = self.xe(logits.view(-1, self.bert_model.config.vocab_size), tags.view(-1))

        # for forced_choice probing tasks
        elif task == 'forced_choice':  # during probing
            logits = self.head_mlm(bert_embeddings)  # projects to vector of size bert_config.vocab_size
            if tags is not None:  # tags must not be of NoneType, because loss must be computed
                # loss function requires that probability distributions are stored in 2nd dim, thus we need to permute
                # logits need to be [batch size, vocab size, seq length]
                # tags need to be [batch size, vocab size]
                loss = self.forced_choice_xe(logits.permute(0, 2, 1), tags)

        # for SRL training
        elif task == 'srl':
            logits = self.head_srl(embedded_text_input)
            if tags is not None:
                loss = sequence_cross_entropy_with_logits(logits, tags, attention_mask)

        else:
            raise AttributeError('Invalid arg to "task"')

        output_dict = {
            'tokens': tokens['tokens'],          # for decoding MLM tags
            'loss': loss,
            "logits": logits,
            "attention_mask": attention_mask,    # for decoding BIO SRL tags
            'start_offsets': [],                 # for decoding BIO SRL tags
            'in': [],                            # for decoding MLM tags
            'gold_tags': [],                     # for computing f1 score
        }

        # add additional info for decoding
        for d in metadata:
            output_dict['start_offsets'].append(d['start_offsets'])
            output_dict['in'].append(d['in'])
            output_dict['gold_tags'].append(d['gold_tags'])

        return output_dict

    def decode_mlm(self,
                   output_dict: Dict[str, Any],
                   ) -> List[List[str]]:
        """
        :returns original sequence with [MASK] replaced with highest scoring word-piece.
        No viterbi or handling word-piece sequences, because task is MLM, not SRL.
        """

        logits = output_dict['logits'].detach().cpu().numpy()
        tokens = output_dict['tokens'].detach().cpu().numpy()  # integer array with shape [batch size, seq length]

        res = []
        num_sequences = len(logits)
        assert num_sequences == len(output_dict['tokens'])
        for seq_id in range(num_sequences):

            # get predicted wp
            wp_id = np.where(tokens[seq_id] == configs.Data.mask_vocab_id)
            assert len(wp_id) == 1
            logits_for_masked_wp = logits[seq_id][wp_id]  # shape is now [vocab_size]
            tag_wp_id = np.asscalar(np.argmax(logits_for_masked_wp))
            tag_wp = self.id2tag_wp_mlm[tag_wp_id]

            # fill in input sequence
            mlm_in = output_dict['in'][seq_id]
            filled_in_sequence = mlm_in.copy()
            filled_in_sequence[mlm_in.index('[MASK]')] = tag_wp
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
        attention_mask = get_lengths_from_binary_sequence_mask(output_dict['attention_mask']).data.tolist()

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        transition_matrix = torch.zeros([len(self.id2tag_wp_srl), len(self.id2tag_wp_srl)])

        # loop over each sequence in batch
        res = []
        for seq_id in range(logits.shape[0]):
            # get max likelihood tags
            length = attention_mask[seq_id]
            tag_wp_probabilities = class_probabilities[seq_id].detach().cpu()[:length]
            ml_tag_wp_ids, _ = viterbi_decode(tag_wp_probabilities, transition_matrix)  # ml = max likelihood
            ml_tags_wp = [self.id2tag_wp_srl[tag_id] for tag_id in ml_tag_wp_ids]

            # convert back from wordpieces
            ml_tags = [ml_tags_wp[i] for i in output_dict['start_offsets'][seq_id]]  # specific to BIO SRL tags
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