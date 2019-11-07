from typing import Dict, List, Optional, Any, Union
import torch
from torch.nn import Linear, Dropout, functional as F
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert import modeling

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.nn.util import viterbi_decode
from allennlp.training.util import rescale_gradients
from allennlp.common import Params as AllenParams
from overrides import overrides


class SrlBert(Model):
    """
        Parameters
        ----------
        vocab : ``Vocabulary``, required
            A Vocabulary, required in order to compute sizes for input/output projections.
        bert_model : ``Union[str, BertModel]``, required.
            A string describing the BERT model to load or an already constructed BertModel.
        initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
            Used to initialize the model parameters.
        regularizer : ``RegularizerApplicator``, optional (default=``None``)
            If provided, will be used to calculate the regularization penalty during training.
        label_smoothing : ``float``, optional (default = 0.0)
            Whether or not to use label smoothing on the labels when computing cross entropy loss.
        """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None) -> None:
        super(SrlBert, self).__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)  # Ph; too big for a single  GTX 1080 Ti
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("labels")

        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.Tensor],
                verb_indicator: torch.Tensor,
                metadata: List[Any],
                tags: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.
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
        verb_indicator = verb_indicator.cuda()
        if tags is not None:
            tags = tags.cuda()

        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(input_ids=tokens["tokens"],
                                             token_type_ids=verb_indicator,
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)

        # TODO test
        print(bert_embeddings.shape)


        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()

        logits = self.tag_projection_layer(embedded_text_input)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])

        # probabilities are defined over word-pieces
        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "mask": mask}

        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ph: Do NOT perform Viterbi decoding - we are interested in learning dynamics, not best performance.

        Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]

        # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
        wordpiece_tags = []
        word_tags = []
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])
        start_transitions = torch.zeros(num_labels)

        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(predictions_list,
                                                sequence_lengths,
                                                output_dict["wordpiece_offsets"]):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix,
                                                        allowed_start_transitions=start_transitions)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]

            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
        output_dict['wordpiece_tags'] = wordpiece_tags
        output_dict['tags'] = word_tags
        return output_dict

    def train_on_batch(self, batch, optimizer):
        """
        written by ph to keep interface between models consistent
        """
        # to cuda
        batch['tokens']['tokens'] = batch['tokens']['tokens'].cuda()
        batch['verb_indicator'] = batch['verb_indicator'].cuda()
        batch['tags'] = batch['tags'].cuda()

        # forward + loss
        optimizer.zero_grad()
        output_dict = self(**batch)  # input is dict[str, tensor]
        loss = output_dict["loss"] + self.get_regularization_penalty()
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        # backward + update
        loss.backward()
        rescale_gradients(self, grad_norm=1.0)
        optimizer.step()

        return loss

    @staticmethod
    def handle_metadata(metadata):
        """
        added by ph.
        moved below code from self.forward() here because it was not used there
        """
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        return list(words), list(verbs), list(offsets)


def make_bertsrl(params,
                 vocab,  # must be Allen Vocabulary instance
                 ):

    print('Preparing BERT model...')
    # parameters of original implementation are specified here:
    # https://github.com/allenai/allennlp/blob/master/training_config/bert_base_srl.jsonnet

    # TODO test
    print(vocab.print_statistics())
    print(vocab._retained_counter)
    raise SystemExit('Debugging')
    vocab_size = vocab.get_vocab_size("tokens")

    config = modeling.BertConfig(vocab_size_or_config_json_file=vocab_size,  # was 32K
                                 hidden_size=params.hidden_size,  # was 768
                                 num_hidden_layers=params.num_layers,  # was 12
                                 num_attention_heads=params.num_attention_heads,  # was 12
                                 intermediate_size=params.intermediate_size)  # was 3072
    bert_model = modeling.BertModel(config=config)

    # TODO bert also needs
    #  * learning rate scheduler - "slanted_triangular"

    # initializer
    initializer_params = [
        ("",
         AllenParams({}))
    ]
    initializer = InitializerApplicator.from_params(initializer_params)

    # model
    model = SrlBert(vocab=vocab,
                    bert_model=bert_model,
                    initializer=initializer,
                    embedding_dropout=0.1)
    model.cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Finished model preparation.'
          'Number of model parameters: {:,}'.format(num_params), flush=True)

    return model