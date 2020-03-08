"""
Parameter exploration notes:

Increasing num_masked doesn't improve dev-f1, but does improve dev-pp.
using num_masked=4 gives markedly better results than any lower values

batch_size=16 gives better devel-pp compared to 32,
and batch_size 32 gives better devel-f1 compared to 128

while delaying SRL to 20K compared to 2K doesn't help development-f1,
it does help development-pp, which is unintuitive, but not impossible.
no delay however, seems best, because it has a large improvement on dev-pp at end of training,
while the reduction in dev-f1 is minimal.

1 MLM epoch is good enough, resulting in best dev-f1.

srl_probability=1.0 results in best dev-f1 than any lower value when MLM epochs = 1.
when lower than 1.0, performance degrades, suggesting strong competition between SRL and MLM tasks.

vocabulary sizes 2K and 16K do not influence end-of-training dev-f1 compared to vocab_size=4K

TODO:
* dropout
* learning rate
* learning rate schedule
* weight smoothing
* regularization ?

"""

param2requests = {
    'embedding_dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
}

# With num_masked=1, made 0,575,465 utterances -> 035,966 train MLM batches (when batch-size=16)
# With num_masked=6, made 2,976,614 utterances -> 186,038 train MLM batches (when batch-size=16)

param2debug = {
    "num_mlm_epochs": 1,
    'num_masked': 1,
}

param2default = {
    'batch_size': 16,  # 16 is slightly better than 32, and 32 is better than 128
    'embedding_dropout': 0.1,  # originally 0.1
    'hidden_size': 128,
    'num_layers': 8,  # 6 is better than any lower number
    'num_attention_heads': 8,
    'intermediate_size': 256,
    'srl_interleaved': True,
    'srl_probability': 1.0,  # any less is worse, any more is unnecessary, even with 1 MLM epoch
    'num_mlm_epochs': 1,
    'num_masked': 4,
    'corpus_name': 'childes-20191206',
    'vocab_size': 4000,  # very robust with respect to dev-f1
}
