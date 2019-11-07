

param2requests = {
    "batch_size": [32],
}

param2debug = {
    "num_epochs": 2,
}

param2default = {
    'batch_size': 32,  # 32 is original implementation
    'hidden_size': 256,
    'num_layers': 2,
    'num_attention_heads': 2,
    'intermediate_size': 512,
    'max_sentence_length': 128,
    'num_epochs': 50,  # 15 fine-tuning epochs is original implementation
}
