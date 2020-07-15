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

vocabulary sizes 2K and 16K do not influence end-of-training dev-f1 compared to vocab_size=4K.

any more dropout than 0.1 worsens dev-pp but does not affect dev-f1.

num_layers < 6 results in worse dev-f1.

more than 0.01 weight_decay results in both worse dev-pp and dev-f1.

when learning rate is 1e-4, performance is as good as but faster than 1e-5, and any lower than 1e-5 is too slow.

performance is very robust with respect to intermediate_size, with 32 being slightly worse than 64

best dev-pp is strongly dependent on large hidden size: 256 is much better than 128 or 64,
and larger hidden sizes speed dev-f1 but do not affect end-of training dev-f1

best hidden size is 256, any lower increases dev-pp

Notes:
    because best performance on both MLM and SRL are achieved when interleaved compared to sequential,
    this suggests that hypothesis space at last layer in BERT is still very unconstrained.

"""

param2requests = {
    'srl_interleaved': [False, True],
    'google_vocab_rule': ['inclusive']
}

# With num_masked=1, made 0,575,465 instances -> 035,966 train MLM batches (when batch-size=16)
# With num_masked=3, made XXXXXXXXX instances -> 107,964 train MLM batches (when batch-size=16)
# With num_masked=6, made 2,976,614 instances -> 186,038 train MLM batches (when batch-size=16)

param2debug = {
    "num_mlm_epochs": 1,
    'num_masked': 1,
    'num_layers': 2,
    'srl_interleaved': False,
    'google_vocab_rule': 'exclusive',
}

param2default = {
    'batch_size': 16,
    'embedding_dropout': 0.1,
    'lr': 1e-4,
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'srl_interleaved': True,
    'srl_probability': 1.0,
    'num_mlm_epochs': 1,
    'num_masked': 6,
    'corpus_name': 'childes-20191206',
    'vocab_size': 4000,
    'google_vocab_rule': 'inclusive',
}
