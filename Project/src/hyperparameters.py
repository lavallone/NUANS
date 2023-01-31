from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "FairySum/texts"
    train_dir: str = "FairySum/texts/train"
    test_dir: str = "FairySum/texts/test"
    texts_path: str = "data/texts.json"
    gold_path: str = "data/gold/gold.json"
    candidates_path: str = "data/candidates/candidates.json"
    scores_path: str = "data/candidates/scores.json"
    abstractives_path: str = "data/abstractives/abstractives.json"
    batch_size: int = 3 # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # EXTRACTION phase params
    sbert_mode: str = "extraction" # extraction or evaluation
    
    # SELECTION phase params
    length_conf_int: int = 3
    k_range: int = 6
    pick_random_n: int = 4
    
    # BERT params
    model: str = "bert" # or "roberta" or "longformer"
    fine_tune: str = "v2"
    hidden_features: int = 768 # don't know if  I'll use it
    max_length: int = 512
    lr: float = 2e-4 # 2e-4 or 1e-3
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    margin_loss: float = 0.01
    max_num_chunks_text: int = 10
    max_num_chunks: int = 2
    
# BERT --> batch=2, num_candidates=51, max_num_chunks_text=10, max_num_chunks=2, fine_tune=v1!
# BERT --> batch=2, num_candidates=51, max_num_chunks_text=10, max_num_chunks=2, fine_tune=v2!

# BERT --> batch=4, num_candidates=25, max_num_chunks_text=10, max_num_chunks=2, fine_tune=v1!
# BERT --> batch=3, num_candidates=25, max_num_chunks_text=10, max_num_chunks=2, fine_tune=v2!
# BERT --> batch=2, num_candidates=25, max_num_chunks_text=10, max_num_chunks=3, fine_tune=v1!