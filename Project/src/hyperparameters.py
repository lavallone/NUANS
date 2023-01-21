from dataclasses import dataclass

@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "FairySum/texts"
    batch_size: int = 64 # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # EXTRACTION phase params
    sbert_mode: str = "extraction" # extraction or evaluation
    
    # SELECTION phase params
    length_conf_int: int = 5
    k_range: int = 10
    pick_random_n: int = 10
    
    # TRANSFORMERS params
    #latent_size: int = 128
    lr: float = 2e-4 # 2e-4 or 1e-3
    #threshold: float = 0.5 # initialization of the threshold
    #gaussian_initialization: bool = True # perform or not the Gaussian inizialization
    #t_weight: float = 0.65 # how much weight the new threshold wrt the old
    #loss_weight: float = 1 # how much weight the reconstruction loss between two pixels 
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    #noise: float = 0.4 # noise factor in (0,1) for the image -- denoising strategy
    #contractive: bool = False # choose if apply contraction to the loss of not
    #lamb: float = 1e-3 # controls the relative importance of the Jacobian (contractive) loss.
    #reduction: str = "mean" # "mean" or "sum" according to the reduction loss strategy
    #slope: float = 0.5 # slope for the leaky relu in convolutions
    
    # LOGGING params
    #log_images: int = 4 # how many images to log each time
    #log_image_each_epoch: int = 2 # epochs interval we wait to log images   