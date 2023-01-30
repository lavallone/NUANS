from collections import Counter
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
import json
import pytorch_lightning as pl
from .data_module import FairySum_DataModule
import random
import gc
from transformers import BertModel, RobertaModel, LongformerModel
from datasets import load_metric


class MatchSum(pl.LightningModule):
    def __init__(self, hparams):
        super(MatchSum, self).__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.model == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif self.hparams.model == "roberta":
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif self.hparams.model == "longformer":
            self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        
        for param in self.model.parameters():
            param.requires_grad = False
        if self.hparams.fine_tune: # da capire meglio quali layers unfreezare o no !!!!
            # we can unfreeze only some layers due to limited gpu card memory --> this is because when we train the tensor 'with gradient' are loaded to the GPU !
            for param in self.model.pooler.parameters():
                param.requires_grad = True
            unfreeze = [10, 11] # the last two layers of the encoder block
            for i in unfreeze:
                for param in self.model.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.rouge = load_metric("rouge")

    def compute_chunk_embedding(self, text_embed, num_chunks):
        init_index = 0
        new_text_embed = []
        for i in num_chunks:
            t = torch.index_select(text_embed, 0, torch.arange(init_index+i).to(text_embed.device))
            new_text_embed.append(t.mean(dim=0))
            init_index += i
        return torch.stack(new_text_embed)
    
    def forward_1(self, text):
        num_chunks = text["num_chunks"]
        chunk_text_embed = self.model(text["input_ids"], attention_mask = text["attention_mask"]).pooler_output
        chunk_text_embed = chunk_text_embed
        text_embed = self.compute_chunk_embedding(chunk_text_embed, num_chunks)
        del text
        del chunk_text_embed
        return text_embed

    def forward_2(self, candidate):
        num_chunks = candidate["num_chunks"]
        chunk_cand_embed = self.model(candidate["input_ids"], attention_mask = candidate["attention_mask"]).pooler_output
        chunk_cand_embed = chunk_cand_embed
        cand_embed = self.compute_chunk_embedding(chunk_cand_embed, num_chunks)
        del candidate
        del chunk_cand_embed
        return cand_embed

    def forward_3(self, gold):
        num_chunks = gold["num_chunks"]
        chunk_gold_embed = self.model(gold["input_ids"], attention_mask = gold["attention_mask"]).pooler_output
        chunk_gold_embed = chunk_gold_embed
        gold_embed = self.compute_chunk_embedding(chunk_gold_embed, num_chunks)
        del gold
        del chunk_gold_embed
        return gold_embed
    
    def forward_4(self, abstractive):
        num_chunks = abstractive["num_chunks"]
        chunk_abstr_embed = self.model(abstractive["input_ids"], attention_mask = abstractive["attention_mask"]).pooler_output
        chunk_abstr_embed = chunk_abstr_embed
        abstr_embed = self.compute_chunk_embedding(chunk_abstr_embed, num_chunks)
        del abstractive
        del chunk_abstr_embed
        return abstr_embed
        
    def forward(self, x):
        text = x["text"]
        text_embed = self.forward_1(text)
  
        # ------------------- candidates ------------------- #
        candidates = []
        candidates_num = len(x["candidates"])
        for i in range(candidates_num):
            candidate = x["candidates"][i]
            cand_embed = self.forward_2(candidate)
            candidates.append(cand_embed)
  
        scores = x["scores"]
        candidates_scores_list = []
        # I need to sort the candidates embeddings according to their ROUGE scores
        t = torch.cat(candidates, dim=1)
        for i in range(t.shape[0]): # i.e. self.hparams.batch_size  
            cand_embed_list = torch.chunk(t[i], candidates_num)
            cand_embed_list_zip = list(zip(scores[i], cand_embed_list))
            cand_embed_list_zip.sort(key=lambda x : x[0], reverse=True)
            sorted_cand_embed_list = [e[1] for e in cand_embed_list_zip]
            sorted_cand_embed_stack = torch.stack(sorted_cand_embed_list)
            new_text_embed = text_embed[i].expand_as(sorted_cand_embed_stack)
            candidates_scores_list.append(torch.cosine_similarity(sorted_cand_embed_stack, new_text_embed.to(sorted_cand_embed_stack.device), dim=-1))
        candidates_scores = torch.stack(candidates_scores_list)
        del scores
        
        if not self.training: # eval mode
            return candidates_scores
        
        else: # train mode
            
            # ------------------- gold ------------------- #
            golds = []
            for i in range(3):
                gold = x["gold"][i]
                gold_embed = self.forward_3(gold)
                golds.append(gold_embed)

            gold_score_list = []
            num_gold = x["num_gold"]
            t = torch.cat(golds, dim=1)
            for i in range(t.shape[0]): # i.e. self.hparams.batch_size  
                gold_embed_list = torch.chunk(t[i], 3) # we consider all the three gold summaries 
                gold_embed_list = gold_embed_list[:num_gold[i]] # we select the real number of gold summaries for the particular story
                gold_embed_stack = torch.stack(gold_embed_list)
                new_text_embed = text_embed[i].expand_as(gold_embed_stack)
                gold_score_list.append(torch.cosine_similarity(gold_embed_stack, new_text_embed.to(gold_embed_stack.device), dim=-1).mean())
            gold_score = torch.tensor(gold_score_list)
            
            # ------------------- abstractive ------------------- #
            abstractive = x["abstractive"]
            abstr_embed = self.forward_4(abstractive)
            abstractive_score = torch.cosine_similarity(abstr_embed, text_embed.to(abstr_embed.device), dim=-1)

            return candidates_scores, gold_score, abstractive_score

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=self.hparams.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self, candidates_scores, gold_score, abstractive_score):
        # the total loss is based on three MarginRanking losses
        #-------------------- candidates/candidates loss -----------------------#
        loss_1 = 0
        for i in range(1, candidates_scores.size(1)):
            pos_score = candidates_scores[:, :-i]
            neg_score = candidates_scores[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size())
            loss = torch.nn.MarginRankingLoss(self.hparams.margin_loss * i)
            loss_1 += loss(pos_score, neg_score, ones)

        #-------------------- candidates/gold loss -----------------------#
        loss_2 = 0
        pos_score = gold_score.unsqueeze(-1).expand_as(candidates_scores)
        neg_score = candidates_scores
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size())
        loss = torch.nn.MarginRankingLoss(0.0)
        loss_2 += loss(pos_score, neg_score, ones)
        #-------------------- candidates/abstractive loss -----------------------#
        loss_3 = 0
        pos_score = abstractive_score.unsqueeze(-1).expand_as(candidates_scores)
        neg_score = candidates_scores
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size())
        loss = torch.nn.MarginRankingLoss(0.0)
        loss_3 += loss(pos_score, neg_score, ones)
        
        return {"loss": loss_1 + loss_2 + loss_3}

    def training_step(self, batch, batch_idx):
        cand_score, gold_score, abstr_score = self(batch)
        loss = self.loss_function(cand_score, gold_score, abstr_score)
        #elf.log_dict(loss)
        # since we only monitor the loss for the training phase, we don't need to write additional 
          # code in the 'training_epoch_end' function!
        return {'loss': loss['loss']}

    def predict(self, batch):
        with torch.no_grad():
            cand_score = self(batch) # if the model is in eval() mode the forward returns only the candidates scores!
            best_candidate_indeces = torch.argmax(cand_score, dim=1) # (1, batch)
            ris = []
            for story, idx in list(zip(batch["id"], best_candidate_indeces)):
                best_candidate_summary = ' '.join([FairySum_DataModule.texts[story][i] for i in FairySum_DataModule.candidates[story][idx]])
                ris.append(best_candidate_summary)
            return ris

    def compute_ROUGE(self, ids, best_candidates):
        rouge_score_list = []
        for story, best_candidate in list(zip(ids, best_candidates)):
            gold_list = FairySum_DataModule.gold_test[story]
            num_gold = len(gold_list)
            story_score = 0
            for g in gold_list:
                gold_summary = ' '.join([FairySum_DataModule.texts[story][i] for i in g])
                results = self.rouge.compute(predictions=[best_candidate], references=[gold_summary])
                story_score += ((results["rougeL"].low.fmeasure+results["rougeL"].mid.fmeasure+results["rougeL"].high.fmeasure)/3)
            story_score /= num_gold
            rouge_score_list.append(story_score)
        return (torch.tensor(rouge_score_list).mean())[0]
 
    def validation_step(self, batch, batch_idx):
        pred = self.predict(batch) # it returns the list of the best summaries for each story
        # good practice to follow with pytorch_lightning for logging values each iteration!
        # https://github.com/Lightning-AI/lightning/issues/4396
        self.compute_ROUGE.update(batch["id"], pred) # speriamo funzioni!
        self.log("val_ROUGE", self.compute_ROUGE, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

    # def validation_epoch_end(self, outputs):