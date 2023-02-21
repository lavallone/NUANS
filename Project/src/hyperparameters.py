from dataclasses import dataclass

@dataclass
class Hparams:
    # EXTRACTION phase params
    sbert_mode: str = "extraction" # extraction or evaluation mode for SBERT model
    
    # SELECTION phase params 
    # "n" is the predicted summary output length
    length_conf_int: int = 5 # confidence interval for the predicted summary length --> n = n + 5
    k_range: int = 10 # how many candidate sets with different length --> the i-th set has binom(n, n-k_range+i) possible candidates
    pick_random_n: int = 10 # how many summaries to sample randomly from each set (k_range sets in total)
    # with this setting will be produced 10*10=100 candidates!
    
    # dataloader params
    dataset_dir: str = "FairySum/texts"
    train_dir: str = "FairySum/texts/train"
    test_dir: str = "FairySum/texts/test"
    texts_path: str = "data/texts.json" # file where each storyis divided in list of sentences
    gold_path: str = "data/gold/gold.json" # file composed by the sentences indices of each gold summary
    candidates_path: str = "data/candidates/candidates_"+str(length_conf_int)+"_"+str(k_range)+"_"+str(pick_random_n)+".json" # file composed by the sentences indices of each candidate summary
    scores_path: str = "data/candidates/scores_"+str(length_conf_int)+"_"+str(k_range)+"_"+str(pick_random_n)+".json" # file which contains the ROUGE-L scores of each candidate summary (computed before training)
    abstractives_path: str = "data/abstractives/abstractives.json" # file which has for each story the generated abstractive summary (with no division in sentences)
    batch_size: int = 2 # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # BERT params
    model: str = "bert" # "bert", "roberta" or "longformer"
    fine_tune: str = "v2" # "v1" or "v2"
    hidden_features: int = 768 # do not change it!
    max_length: int = 512 # 512 for Bert and Roberta and 4096 for LongFormer
    cls: int = 101 # 101 for Bert, 0 for RoBERTa and for LongFormer
    sep: int = 102 # 102 for Bert, 2 for RoBERTa and for LongFormer
    lr: float = 1e-4 # 1e-4 or 1e-3
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    max_num_chunks_text: int = 10 # maximum number of chunks for the original texts
    max_num_chunks: int = 4 # maximum number of chunks for the gold, candidate and abstractive summaries