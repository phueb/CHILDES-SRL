from typing import Dict, List, Any
import torch
from torch.nn import Linear
from torch.nn import CrossEntropyLoss

from childes_srl.utils import sequence_cross_entropy_with_logits


class BertForMLMAndSRL(torch.nn.Module):
    """
    Multi-task BERT.
    It has a head for MLM and another head for SRL, and can be trained jointly on both objectives
    """

    def __init__(self,
                 bert_encoder: Any,
                 num_tags_mlm: int,
                 num_tags_srl: int,
                 ignore_token_id: int,
                 ) -> None:

        super().__init__()

        self.bert_encoder = bert_encoder  # TODO implement, e.g. hugginface transformers.BertModel

        # make one BERT head for MLM, and SRL
        self.head_mlm = Linear(self.bert_model.config.hidden_size, num_tags_mlm)
        self.head_srl = Linear(self.bert_model.config.hidden_size, num_tags_srl)

        # one loss function for each objective
        self.xe = CrossEntropyLoss(ignore_index=ignore_token_id)

    def forward(self,
                task: str,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,  # indicates position of predicate when task == 'srl'
                attention_mask: torch.Tensor,
                tags: torch.LongTensor = None,
                use_gpu: bool = True,
                ) -> Dict[str, torch.Tensor]:

        loss = None

        # move to GPU
        if use_gpu:
            input_ids.to('cuda')
            token_type_ids.to('cuda')
            if tags is not None:
                tags.to('cuda')

        # get BERT contextualized embeddings - modeled after huggingface transformers package, dummy code
        outputs = self.bert_encoder(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask
                                    )
        bert_embeddings = outputs[0]

        # for MLM training
        if task == 'mlm':
            logits = self.head_mlm(bert_embeddings)  # projects to vector of size bert_config.vocab_size
            if tags is not None:
                loss = self.xe(logits.view(-1, self.bert_model.config.vocab_size), tags.view(-1))

        # for SRL training
        elif task == 'srl':
            logits = self.head_srl(bert_embeddings)
            if tags is not None:
                loss = sequence_cross_entropy_with_logits(logits, tags, attention_mask)

        else:
            raise AttributeError('Invalid arg to "task"')

        output = {
            'loss': loss,
            "logits": logits,
        }

        return output