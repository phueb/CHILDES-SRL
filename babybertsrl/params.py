

param2requests = {
    'num_pre_train_epochs': [0, 1],
}

param2debug = {
    "num_pre_train_epochs": 0,
    "num_fine_tune_epochs": 1,
    'num_masked': 1,
}

param2default = {
    'batch_size': 256,  # 32 is original implementation
    'hidden_size': 128,
    'num_layers': 8,  # 6 is better than any lower number
    'num_attention_heads': 8,
    'intermediate_size': 256,
    'num_pre_train_epochs': 1,
    'num_fine_tune_epochs': 1,
    'num_masked': 7,
    'corpus_name': 'childes-20191206',
    'vocab_size': 4000,
}
