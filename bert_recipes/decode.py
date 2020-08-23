"""
Code obtained from Allen AI NLP toolkit in September 2019
Modified by PH March 2020
"""
from typing import List, Dict, Any
import numpy as np
import torch
from torch.nn import functional as F

from childes_srl.utils import get_lengths_from_binary_sequence_mask


def decode_mlm_batch_output(token_ids: torch.tensor,  # integer array with shape [batch size, seq length]
                            logits: torch.tensor,
                            utterances: List[List[str]],
                            mask_token_id: int,  # token_id corresponding to [MASK]
                            ) -> List[List[str]]:
    """
    :returns original utterance with [MASK] replaced with highest scoring word-piece.
    """

    logits = logits.detach().cpu().numpy()
    token_ids = token_ids.detach().cpu().numpy()

    res = []
    num_sequences = len(logits)
    assert num_sequences == len(token_ids)

    for seq_id in range(num_sequences):
        # get predicted wp
        wp_id = np.where(token_ids[seq_id] == mask_token_id)
        assert len(wp_id) == 1
        logits_for_masked_wp = logits[seq_id][wp_id]  # shape is now [vocab_size]
        tag_wp_id = np.asscalar(np.argmax(logits_for_masked_wp))
        tag_wp = id2mlm_tag[tag_wp_id]

        # fill in input sequence
        mlm_in = utterances[seq_id]
        filled_in_sequence = mlm_in.copy()
        filled_in_sequence[mlm_in.index('[MASK]')] = tag_wp
        res.append(filled_in_sequence)

    return res  # sequence with predicted word-piece, one per sequence in batch


def decode_srl_batch_output(logits: torch.tensor,
                            start_offsets: List[List[int]],
                            attention_mask: torch.tensor,
                            id2srl_tag: Dict[int, str],
                            ) -> List[List[str]]:
    """
    for each sequence in batch:
    1) get max likelihood tags
    2) convert back from wordpieces


    Note: no decoding constraints imposed during viterbi decoding

    Note: decoding is performed on word-pieces, and word-pieces are then converted to whole words
    """

    # get probabilities
    reshaped_logits = logits.view(-1, len(id2srl_tag))  # collapse time steps and batches
    class_probabilities = F.softmax(reshaped_logits, dim=-1).view([logits.shape[0],
                                                                   logits.shape[1],
                                                                   len(id2srl_tag)])
    attention_mask = get_lengths_from_binary_sequence_mask(attention_mask).data.tolist()

    # ph: transition matrices contain only ones (and no -inf, which would signal illegal transition)
    transition_matrix = torch.zeros([len(id2srl_tag), len(id2srl_tag)])

    # loop over each sequence in batch
    res = []
    for seq_id in range(logits.shape[0]):
        # get max likelihood tags
        length = attention_mask[seq_id]
        tag_wp_probabilities = class_probabilities[seq_id].detach().cpu()[:length]
        ml_tag_wp_ids, _ = viterbi_decode(tag_wp_probabilities, transition_matrix)  # ml = max likelihood
        ml_tags_wp = [id2srl_tag[tag_id] for tag_id in ml_tag_wp_ids]

        # convert back from wordpieces
        ml_tags = [ml_tags_wp[i] for i in start_offsets[seq_id]]  # specific to BIO SRL tags
        res.append(ml_tags)

    return res  # list of max likelihood tags
