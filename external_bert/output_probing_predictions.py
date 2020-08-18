from pathlib import Path
from collections import OrderedDict
from transformers import WordpieceTokenizer, BertConfig, BertForPreTraining
import torch

from transformers import load_tf_weights_in_bert

from allennlp.data.iterators import BucketIterator

from babybertsrl.params import param2default
from babybertsrl.converter import ConverterMLM
from babybertsrl.model import MTBert
from babybertsrl import configs
from babybertsrl.eval import get_probing_predictions


def load_words_from_vocab_file(vocab_file: Path,
                               col: int = 0):

    res = []
    with vocab_file.open("r", encoding="utf-8") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            token = line.split()[col]

            # exclude word with non-ASCII characters
            if [True for c in token if ord(c) > 127]:
                continue

            res.append(token)
    return res


def make_wordpiece_tokenizer(childes_vocab_file: str,
                             google_vocab_file: str):
    """
    uses google vocab as starting point,
    but also adds any word from 4k most frequent words in CHILDES that is not in google vocab
    """

    childes_vocab = load_words_from_vocab_file(Path(childes_vocab_file), col=1)[:4000]
    google_vocab = load_words_from_vocab_file(Path(google_vocab_file), col=0)
    # make vocab index for tokenizer
    to_index = configs.Data.special_symbols + configs.Data.childes_symbols + list(set(childes_vocab + google_vocab))
    vocab = OrderedDict()
    index = 0
    for token in to_index:
        if token in vocab:
            # happens for symbols
            continue
        vocab[token] = index
        index += 1
    assert vocab['[PAD]'] == 0
    assert vocab['[UNK]'] == 1
    assert vocab['[CLS]'] == 2
    assert vocab['[SEP]'] == 3
    assert vocab['[MASK]'] == 4

    return WordpieceTokenizer(vocab, unk_token='[UNK]')


# def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file):  # TODO use or remove
#     # Initialise PyTorch model
#     config = BertConfig.from_json_file(bert_config_file)
#     print("Building PyTorch model from configuration: {}".format(str(config)))
#     model = BertForPreTraining(config)
#
#     # Load weights from tf checkpoint
#     load_tf_weights_in_bert(model, config, tf_checkpoint_path)
#
#     return model


if __name__ == '__main__':

    # path to probing data - probing data can be found at https://github.com/phueb/Babeval/tree/master/sentences
    probing_path = Path().home() / 'Babeval_phueb' / 'sentences'

    google_vocab_file = "../data/bert-base-uncased-vocab.txt"
    childes_vocab_file = '../data/childes-20191206_vocab.txt'
    wordpiece_tokenizer = make_wordpiece_tokenizer(childes_vocab_file, google_vocab_file)

    converter_mlm = ConverterMLM(param2default['num_masked'], wordpiece_tokenizer)

    bucket_batcher_mlm_large = BucketIterator(batch_size=512, sorting_keys=[('tokens', "num_tokens")])

    for path_to_bin in (configs.Dirs.external_bert / 'pretrained_models').glob('*/*.bin'):
        architecture_name = path_to_bin.parent
        bert_config_file = configs.Dirs.external_bert / 'pretrained_models' / architecture_name / 'bert_config.json'
        bin_file = configs.Dirs.external_bert / 'pretrained_models' / path_to_bin

        # load bert model
        config = BertConfig.from_json_file(bert_config_file)
        print("Building PyTorch model from configuration: {}".format(str(config)))
        bert_fpt = BertForPreTraining(config)
        bert_fpt.load_state_dict(torch.load(bin_file))

        # make multi-tasking bert model with bert as base
        mt_bert = MTBert(id2tag_wp_mlm={i: t for t, i in wordpiece_tokenizer.vocab.items()},
                         id2tag_wp_srl={},  # not needed for probing
                         bert_model=bert_fpt.bert,  # BertForPreTraining has object 'bert' which is needed here
                         embedding_dropout=param2default['embedding_dropout'])
        mt_bert.cuda()
        num_params = sum(p.numel() for p in mt_bert.parameters() if p.requires_grad)
        print('Number of parameters: {:,}\n'.format(num_params), flush=True)

        # CHKPT_NAME = 'test.ckpt'  # TODO
        #
        # # a path or url to a pretrained model archive containing:
        # # 1) bert_config.json or openai_gpt_config.json a configuration file for the model, and
        # # 2) pytorch_model.bin a PyTorch dump of a pre-trained instance of
        # # BertForPreTraining, OpenAIGPTModel, TransfoXLModel, GPT2LMHeadModel (saved with the usual torch.save())
        # tf_checkpoint_path = configs.Dirs.external_bert / 'pretrained_models' / path / CHKPT_NAME
        # bert_config_file = configs.Dirs.external_bert / 'pretrained_models' / path / 'bert_config.json'
        #
        # # load model from tensorflow checkpoint
        # model = convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file)

        save_path = configs.Dirs.external_bert / 'pretrained_models' / architecture_name / 'saves'
        get_probing_predictions(probing_path, converter_mlm, bucket_batcher_mlm_large, save_path, mt_bert)
