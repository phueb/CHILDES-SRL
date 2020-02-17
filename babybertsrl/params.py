

param2requests = {
    'srl_task_delay': [0, 20_000],  # TODO
    'srl_task_ramp': [20_000],  # TODO
}

param2debug = {
    "num_srl_epochs": 1,
    "num_mlm_epochs": 1,
    'num_masked': 1,
}

param2default = {
    'batch_size': 128,  # 32 is original implementation
    'hidden_size': 128,
    'num_layers': 8,  # 6 is better than any lower number
    'num_attention_heads': 8,
    'intermediate_size': 256,
    'srl_task_delay': 0,  # number of steps to wait before training on srl task
    'srl_task_ramp': 1_000_000,  # number of steps during which probability of srl training increases
    'num_srl_epochs': 1,
    'num_mlm_epochs': 1,
    'num_masked': 7,
    'corpus_name': 'childes-20191206',
    'vocab_size': 4000,
}
