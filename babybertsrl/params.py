

param2requests = {
    "num_layers": [8],
    "num_attention_heads": [8],
    "hidden_size": [512],
    "intermediate_size": [1024],
}

param2debug = {
    "num_epochs": 2,
}

param2default = {
    'batch_size': 32,  # 32 is original implementation
    'hidden_size': 256,
    'num_layers': 8,  # 6 is better than any lower number
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'max_sentence_length': 128,
    'num_epochs': 50,  # 15 fine-tuning epochs is original implementation
}
