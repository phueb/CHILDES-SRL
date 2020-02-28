"""
Parameter exploration notes:

Increasing num_masked doesn't improve dev-f1, but does slightly improve dev-pp.

batch_size=16 gives better devel-pp compared to 32,
and batch_size 32 gives better devel-f1 compared to 128

so far the best f1 on human-based_2018 is 0.77, which happens around step 60K,
corresponding to no more than 2 SRL training epochs

while delaying SRL to 20K compared to 2K doesn't help development-f1,
it does help development-pp, which is unintuitive, but not impossible

2 MLM epochs are better than 1.

2 SRL epochs improve dev-pp, but only with a small SRL delay (no more than 2K0,
otherwise there is too much SRL training at end, and MLM knowledge is forgotten (increase in dev-pp)


TODO:
* vocab size
* dropout
* learning rate
* learning rate schedule
* weight smoothing
* regularization ?

"""

param2requests = {
    'srl_task_delay': [2_000],  # TODO
    'srl_task_ramp': [0],
    'num_masked': [5, 6, 7, 8],  # it seems, the lower the better dev-pp, surprisingly
    'num_srl_epochs': [2],
    'num_mlm_epochs': [2],
}

param2debug = {
    "num_srl_epochs": 1,
    "num_mlm_epochs": 1,
    'num_masked': 1,
}

param2default = {
    'batch_size': 16,  # 16 is slightly better than 32, and 32 is better than 128
    'hidden_size': 128,
    'num_layers': 8,  # 6 is better than any lower number
    'num_attention_heads': 8,
    'intermediate_size': 256,
    'srl_task_delay': 0,  # number of steps to wait before training on srl task
    'srl_task_ramp': 0,  # number of steps during which probability of srl training increases
    'num_srl_epochs': 1,
    'num_mlm_epochs': 1,
    'num_masked': 7,
    'corpus_name': 'childes-20191206',
    'vocab_size': 4000,
}
