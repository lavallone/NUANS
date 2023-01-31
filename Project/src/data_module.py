import os
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import json
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer
from collections import Counter
import torch
from functools import partial
from src.hyperparameters import Hparams
from dataclasses import asdict
from tqdm import tqdm
import random

def create_chunk_tokens(tokens, max_encoder_length, max_num_chunks, max_num_chunks_text=None):
    if max_num_chunks_text != None: # it means we're chunking the original text
        max_num_chunks = max_num_chunks_text
    
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # I'm going to need this cleaning phase because if I want to work with batches and I have input texts of different lengths,
    # the tokenizer without the padding set to True gave me an error.
    # So we set padding=True and we now clean the not needed 0s in addition! 
    mask = attention_mask.bool()
    input_ids_list = []
    attention_mask_list = []
    for i in range(input_ids.shape[0]):
        input_ids_list.append(torch.masked_select(input_ids[i], mask[i]))
        attention_mask_list.append(torch.masked_select(attention_mask[i], mask[i]))

    # To be more efficient, I need to implement the possibility of  working with tokens length that are
    # less than the maximum allowed (512 for Bert and Roberta and 4096 for LongFormer)
    max_batch_length = 0
    max_length = torch.tensor([len(e) for e in input_ids_list]).max()
    if max_length >= max_encoder_length-2: 
        max_batch_length = max_encoder_length-2
    else:
        max_batch_length = max_length

    num_chunks_list = []
    ids_stack = []
    attn_stack = []
    for i in range(input_ids.shape[0]): # we itearate through each texts
        ids_chunks = list(input_ids_list[i].split(max_batch_length))
        attn_chunks = list(attention_mask_list[i].split(max_batch_length))
        num_chunks = len(ids_chunks)
        
        if num_chunks > max_num_chunks: # we set a maximum number of chunks for each text
            num_chunks = max_num_chunks
        
        num_chunks_list.append(num_chunks) # in this way in the forward function we know which embedding chunks belong to which text
        for i in range(len(ids_chunks)): # for each created chunk (without considering yet the max number of chunks we set)
            ids_chunks[i] = torch.cat([torch.tensor([101]), ids_chunks[i], torch.tensor([102])])
            attn_chunks[i] = torch.cat([torch.tensor([1]), attn_chunks[i], torch.tensor([1])])
            
            pad_len = (max_batch_length + 2) - len(ids_chunks[i])
            if pad_len > 0:
                ids_chunks[i] = torch.cat([ids_chunks[i], torch.tensor([0] * pad_len)])
                attn_chunks[i] = torch.cat([attn_chunks[i], torch.tensor([0] * pad_len)])
        
        # we select 'num_chunks' randomly 
        rnd_idx = sorted(random.sample(range(len(ids_chunks)), num_chunks))
        ids_stack += [ids_chunks[i] for i in rnd_idx]
        attn_stack += [attn_chunks[i] for i in rnd_idx]

    return {"input_ids" : (torch.stack(ids_stack)).long() , "attention_mask" : (torch.stack(attn_stack)).int(), "num_chunks" : torch.tensor(num_chunks_list).int()}

class FairySum_Dataset(Dataset):
    def __init__(self, data_dir: str, texts_path: str, candidates_path: str, train_or_test: str, gold_path: str = None, scores_path: str = None, abstractives_path: str = None):
        self.data = list()
        self.train_or_test = train_or_test
        self.texts = json.load(open(texts_path, "r"))
        self.candidates = json.load(open(candidates_path, "r"))
        if self.train_or_test == "train":
            self.data_dir = data_dir
            self.gold = json.load(open(gold_path, "r"))
            self.scores = json.load(open(scores_path, "r"))
            self.abstractives = json.load(open(abstractives_path, "r"))
        else:
            self.data_dir = data_dir
        self.make_data()
    
    def make_data(self):
        legal_keys = []
        for f in os.listdir(self.data_dir):
            k = "_".join(f.split("_")[:2])
            legal_keys.append(k)
            
        num_gold_counter = Counter()
        num_gold_counter.update([f[:-12] for f in os.listdir("FairySum/gold")])
        
        for story in self.texts.keys():
            if story not in legal_keys:
                continue
            text = ' '.join(self.texts[story])
            candidates = []
            for c in self.candidates[story]:
                candidates.append(' '.join([self.texts[story][i] for i in c]))
                
            assert self.train_or_test=="train" or self.train_or_test=="test"
            if self.train_or_test == "test":
                scores = [0 for _ in range(51)]
                gold = ["","",""]
                num_gold = 0
                abstractive = ""
            elif self.train_or_test == "train": 
                scores = self.scores[story]
                # ------------------- gold summaries handling ------------------- #
                gold = []
                last_gold = ""
                for g in self.gold[story]:
                    last_gold = ' '.join([self.texts[story][i] for i in g])
                    gold.append(last_gold)
                num_gold = num_gold_counter[story]
                if num_gold == 1:
                    gold += [last_gold, last_gold]
                elif num_gold == 2:
                    gold.append(last_gold)
                assert len(gold) == 3
                # --------------------------------------------------------------- #
                abstractive = self.abstractives[story]
            self.data.append({"id" : story, "text": text, "candidates" : candidates, "scores" : scores, "gold" : gold, "num_gold" : num_gold, "abstractive" : abstractive})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FairySum_DataModule(pl.LightningDataModule):
    # STATIC OBJECTS --> I need to call them for the 'val_ROUGE' computation during the training process
    texts = json.load(open(asdict(Hparams())["texts_path"], "r"))
    candidates = json.load(open(asdict(Hparams())["candidates_path"], "r"))
    gold_test = json.load(open("data/gold/gold_test.json", "r"))
    
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams, logger=False) 

    def setup(self, stage = None):
        if not hasattr(self, "data_train"):
            # TRAIN
            self.data_train = FairySum_Dataset(data_dir=self.hparams.train_dir, texts_path=self.hparams.texts_path, 
                                               candidates_path=self.hparams.candidates_path, train_or_test="train", 
                                               gold_path=self.hparams.gold_path, scores_path=self.hparams.scores_path, abstractives_path=self.hparams.abstractives_path)
            # TEST
            self.data_test = FairySum_Dataset(data_dir=self.hparams.test_dir, texts_path=self.hparams.texts_path, 
                                              candidates_path=self.hparams.candidates_path, train_or_test="test")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )
    
    # for efficiency reasons, each time we pick a batch from the dataloader, we call this function!
    def collate(self, batch):
        batch_out = dict()
        batch_out["id"] = [sample["id"] for sample in batch]
        if self.hparams.model == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.hparams.model == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        elif self.hparams.model == "longformer":
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        batch_out["text"] = create_chunk_tokens( tokenizer([sample["text"] for sample in batch], add_special_tokens=False, padding=True, return_tensors="pt"), self.hparams.max_length, self.hparams.max_num_chunks, self.hparams.max_num_chunks_text )
        candidates_num = len(batch[0]["candidates"])
        batch_out["candidates"] = [ create_chunk_tokens( tokenizer([sample["candidates"][i] for sample in batch], add_special_tokens=False, padding=True, return_tensors="pt"), self.hparams.max_length, self.hparams.max_num_chunks ) for i in range(candidates_num) ]
        batch_out["scores"] = torch.as_tensor([sample["scores"] for sample in batch])
        batch_out["gold"] = [ create_chunk_tokens( tokenizer([sample["gold"][i] for sample in batch], add_special_tokens=False, padding=True, return_tensors="pt"), self.hparams.max_length, self.hparams.max_num_chunks ) for i in range(3) ] # the gold summaries can be maximum 3!
        batch_out["num_gold"] = [sample["num_gold"] for sample in batch]
        batch_out["abstractive"] = create_chunk_tokens( tokenizer([sample["abstractive"] for sample in batch], add_special_tokens=False, padding=True, return_tensors="pt"), self.hparams.max_length, self.hparams.max_num_chunks )
        return batch_out