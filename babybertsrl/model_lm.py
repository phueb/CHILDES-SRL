from typing import Dict, List, Optional, Any, Union
import torch
from torch.nn import Linear, Dropout, functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from overrides import overrides

from pytorch_pretrained_bert.modeling import BertOnlyMLMHead

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients
from allennlp.common import Params as AllenParams


class LMBert(Model):
    """
    custom Model built on top of un-trained Bert to train Bert on masked language modeling task
    """

    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: BertModel,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 ) -> None:
        super(LMBert, self).__init__(vocab, regularizer)

        self.bert_model = bert_model

        # labels namespace is 2 elements shorter than tokens because it does not have PADDING and UNKNOWN
        self.num_out = self.vocab.get_vocab_size('labels')                    # 4099
        self.projection_layer = Linear(self.bert_model.config.hidden_size, self.num_out)

        # NOTE: the BertOnlyMLMHead does not work because the output size must be 2 elements larger because
        # output size is size of allen vocab object and not number of bert word embeddings
        # self.projection_layer = BertOnlyMLMHead(self.bert_model.config,
        #                                         self.bert_model.embeddings.word_embeddings.weight)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.Tensor],
                mask_indicator: torch.Tensor,
                metadata: List[Any],
                lm_tags: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        mask_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the masked-token (ph)
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no mask.
        lm_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata contains the original lm_in in the sentence, the language modeling mask,
             and start offsets for converting wordpieces back to a sequence of lm_in,
            under 'lm_in', 'lm_mask' and 'offsets' keys, respectively.
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

        # added by ph
        tokens['tokens'] = tokens['tokens'].cuda()
        mask_indicator = mask_indicator.cuda()
        if lm_tags is not None:
            lm_tags = lm_tags.cuda()

        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(input_ids=tokens['tokens'],
                                             token_type_ids=mask_indicator,  # indices of tokens to be predicted
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()

        logits = self.projection_layer(embedded_text_input)
        reshaped_log_probs = logits.view(-1, self.num_out)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_out])

        # probabilities are defined over word-pieces
        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "mask": mask}

        # We add in the offsets here so we can compute the un-wordpieced lm_tags.
        lm_in, gold_lm_tags, offsets = zip(*[(x['lm_in'], x['gold_lm_tags'], x['offsets']) for x in metadata])
        output_dict['lm_in'] = list(lm_in)
        output_dict['gold_lm_tags'] = list(gold_lm_tags)
        output_dict['wordpiece_offsets'] = list(offsets)

        # TODO the correct way to do language modeling would be to use
        #  standard cross entropy rather than sequence cross entropy because only one masked word is predicted,
        #  not a sequence of tags

        # TODO use the mask to mask all words which should not be predicted?

        if lm_tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      lm_tags,
                                                      mask)
            output_dict['loss'] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        ph: Do NOT use decoding constraints - transition matrix has zeros only
        we are interested in learning dynamics, not best performance.
        Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
        """

        # get probabilities
        all_predictions = output_dict['class_probabilities']
        predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict['mask']).data.tolist()

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        transition_matrix = torch.zeros([self.num_out, self.num_out])

        # decode
        wordpiece_tags = []
        word_tags = []
        for predictions, length, offsets in zip(predictions_list,
                                                sequence_lengths,
                                                output_dict['wordpiece_offsets']):
            tag_probabilities = predictions[:length]
            max_likelihood_tag_ids, _ = viterbi_decode(tag_probabilities,
                                                       transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_tag_ids]
            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])

        # collect results
        output_dict['wordpiece_lm_tags'] = wordpiece_tags
        output_dict['lm_tags'] = word_tags
        return output_dict

    def train_on_batch(self, batch, optimizer):
        # to cuda
        batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
        batch['mask_indicator'] = batch['mask_indicator'].cuda()
        batch['lm_tags'] = batch['lm_tags'].cuda()

        # forward + loss
        optimizer.zero_grad()
        output_dict = self(**batch)  # input is dict[str, tensor]
        loss = output_dict['loss'] + self.get_regularization_penalty()
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        # backward + update
        loss.backward()
        rescale_gradients(self, grad_norm=1.0)
        optimizer.step()

        return loss


def make_bert_lm(params,
                 data,
                 ):

    print('Preparing BERT model...')
    # parameters of original implementation are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/bert_base_srl.jsonnet

    vocab_size = len(data.bert_tokenizer.vocab)

    config = BertConfig(vocab_size_or_config_json_file=vocab_size,  # was 32K
                        hidden_size=params.hidden_size,  # was 768
                        num_hidden_layers=params.num_layers,  # was 12
                        num_attention_heads=params.num_attention_heads,  # was 12
                        intermediate_size=params.intermediate_size)  # was 3072
    bert_model = BertModel(config=config)

    # TODO bert also needs
    #  * learning rate scheduler - "slanted_triangular"

    # initializer
    initializer_params = [
        ("",
         AllenParams({}))
    ]
    initializer = InitializerApplicator.from_params(initializer_params)

    # model
    model = LMBert(vocab=data.vocab,
                   bert_model=bert_model,
                   initializer=initializer,
                   embedding_dropout=0.1)
    model.cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Finished model preparation.'
          'Number of model parameters: {:,}'.format(num_params), flush=True)

    return model