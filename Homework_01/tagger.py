import re
from transformers import BertTokenizer, BertModel 
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utilities import crf
#import booknlp.common.sequence_eval as sequence_eval

class NER_Model(nn.Module): # the actual model which performs NER tagging

	def __init__(self, freeze_bert=False, base_model=None, tagset=None, tagset_flat=None, supersense_tagset=None, hidden_dim=100, flat_hidden_dim=200, device=None):
		super(NER_Model, self).__init__()

		modelName=base_model
		modelName=re.sub("^entities_", "", modelName)
		modelName=re.sub("-v\d.*$", "", modelName)
		matcher=re.search(".*-(\d+)_H-(\d+)_A-.*", modelName)
		bert_dim=0
		modelSize=0
		self.num_layers=0
		if matcher is not None:
			bert_dim=int(matcher.group(2))
			self.num_layers=min(4, int(matcher.group(1)))

			modelSize=self.num_layers*bert_dim

		assert bert_dim != 0
		
		self.tagset=tagset
		self.tagset_flat=tagset_flat # da capire cosa è

		self.device=device
		self.crf=crf.CRF(len(self.tagset), device)

		self.wn_embedding = nn.Embedding(50, 20)

		# "revised tagset"
		self.rev_tagset={tagset[v]:v for v in tagset}
		self.rev_tagset[len(tagset)] = "O"
		self.rev_tagset[len(tagset)+1] = "O"

		self.num_labels=len(tagset) + 2

		###########################################################################################################################
		## supersense task components --> we keep these fields because the pretrained models have them (otherwise I'd delete them)
		self.supersense_tagset = supersense_tagset
		self.num_supersense_labels = len(supersense_tagset) + 2
		self.supersense_crf = crf.CRF(len(supersense_tagset), device)
		self.rev_supersense_tagset = {supersense_tagset[v]:v for v in supersense_tagset}
		self.rev_supersense_tagset[len(supersense_tagset)]="O"
		self.rev_supersense_tagset[len(supersense_tagset)+1]="O"
  
		self.supersense_lstm1 = nn.LSTM(modelSize + 20, hidden_dim, bidirectional=True, batch_first=True)
		self.supersense_hidden2tag1 = nn.Linear(hidden_dim * 2, self.num_supersense_labels)
		###########################################################################################################################
  
		########################################################################################################
		# BERT
		self.tokenizer = BertTokenizer.from_pretrained(modelName, do_lower_case=False, do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained(modelName)
		self.tokenizer.add_tokens(["[CAP]"], special_tokens=True) # we need it because we use pretrained UNCASED BERT models!
		self.bert.resize_token_embeddings(len(self.tokenizer))
		self.bert.eval()
		if freeze_bert: # we'll use the model as it is!
					for param in self.bert.parameters():
						param.requires_grad = False
		########################################################################################################

		self.hidden_dim = hidden_dim
		self.layered_dropout = nn.Dropout(0.20)

		########################################################################################################
		# 3 LSTMs needed for predicting NESTED NER entities
		self.lstm1 = nn.LSTM(modelSize, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden2tag1 = nn.Linear(hidden_dim * 2, self.num_labels)

		self.lstm2 = nn.LSTM(2*hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden2tag2 = nn.Linear(hidden_dim * 2, self.num_labels)

		self.lstm3 = nn.LSTM(2*hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
		self.hidden2tag3 = nn.Linear(hidden_dim * 2, self.num_labels)
		########################################################################################################

		# we don't need these components for NER tagging
		self.num_labels_flat=len(tagset_flat)
		self.flat_dropout = nn.Dropout(0.5)
		self.flat_hidden_dim=flat_hidden_dim
		self.flat_lstm = nn.LSTM(modelSize, self.flat_hidden_dim, bidirectional=True, batch_first=True, num_layers=1)
		self.flat_classifier = nn.Linear(2*self.flat_hidden_dim, self.num_labels_flat)

	# the usual forward function of 'nn.Module' --> but in our case we never use it!
	def forward(self, input_ids, matrix1, matrix2, attention_mask=None, transforms=None, labels=None, lens=None):
		
		matrix1=matrix1.to(self.device)
		matrix2=matrix2.to(self.device)
		
		input_ids = input_ids.to(self.device)
		attention_mask = attention_mask.to(self.device)
		transforms = transforms.to(self.device)

		if lens is not None:
			lens[0] = lens[0].to(self.device)
			lens[1] = lens[1].to(self.device)
			lens[2] = lens[2].to(self.device)

		if labels is not None:
			labels[0] = labels[0].to(self.device)
			labels[1] = labels[1].to(self.device)
			labels[2] = labels[2].to(self.device)
		
		output = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states=True)
		hidden_states=output["hidden_states"]
		if self.num_layers == 4:
			all_layers = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), 2)
		elif self.num_layers == 2:
			all_layers = torch.cat((hidden_states[-1], hidden_states[-2]), 2)

		# remove the opening [CLS]
		reduced=torch.matmul(transforms,all_layers)[:,1:,:]

		reduced=self.layered_dropout(reduced)

		lstm_out1, _ = self.lstm1(reduced)
		tag_space1 = self.hidden2tag1(lstm_out1)

		input2=torch.matmul(matrix1[:,1:,1:],lstm_out1)

		input2=self.layered_dropout(input2)

		lstm_out2, _ = self.lstm2(input2)
		tag_space2 = self.hidden2tag2(lstm_out2)
		
		input3=torch.matmul(matrix2[:,1:,1:],lstm_out2)

		input3=self.layered_dropout(input3)
		
		lstm_out3, _ = self.lstm3(input3)
		tag_space3 = self.hidden2tag3(lstm_out3)	

		to_value=0

		forward_score1 = self.crf.forward(tag_space1, lens[0]-2)
		sequence_score1 = self.crf.score(torch.where(labels[0][:,1:] == -100, torch.ones_like(labels[0][:,1:]) * to_value, labels[0][:,1:]), lens[0]-2, logits=tag_space1)
		loss1 = (forward_score1 - sequence_score1).sum()

		forward_score2 = self.crf.forward(tag_space2, lens[1]-2)
		sequence_score2 = self.crf.score(torch.where(labels[1][:,1:] == -100, torch.ones_like(labels[1][:,1:]) * to_value, labels[1][:,1:]), lens[1]-2, logits=tag_space2)
		loss2 = (forward_score2 - sequence_score2).sum()

		forward_score3 = self.crf.forward(tag_space3, lens[2]-2)
		sequence_score3 = self.crf.score(torch.where(labels[2][:,1:] == -100, torch.ones_like(labels[2][:,1:]) * to_value, labels[2][:,1:]), lens[2]-2, logits=tag_space3)
		loss3 = (forward_score3 - sequence_score3).sum()

		return loss1 + loss2 + loss3

	def predict_all(self, input_ids, attention_mask=None, transforms=None, lens=None, doEntities=True):

		def fix(sequence):

			"""
			Ensure tag sequence is BIO-compliant

			"""
			for idx, tag in enumerate(sequence):
				tag=self.rev_tagset[tag]
				if tag.startswith("I-"):
					parts=tag.split("-")
					label=parts[1]
					flag=False
					for i in range(idx-1, -1, -1):
						prev=self.rev_tagset[sequence[i]].split("-")

						if prev[0] == "B" and prev[1] == label:
							flag=True
							break
						
						if prev[0] == "O":
							break

						if prev[0] != "O" and prev[1] != label:
							break

					if flag==False:
						sequence[idx]=self.tagset["B-%s" % label]


		def get_layer_transformation(tag_space, t):

			"""
			After predicting a tag sequence, get the information we need to transform the current layer
			to the next layer (e.g., merging tokens in the same entity and remembering which ones we merged)

			"""

			nl=tag_space.shape[1]

			all_tags=[]
			for tags in t:
				all_tags.append(list(tags.data.cpu().numpy()))

			# matrix for merging tokens in layer n+1 that are part of the same entity in layer n 
			all_index=[]
			# indices of tokens that were merged (so we can restored them later)
			all_missing=[]
			# length of the resulting layer (after merging)
			all_lens=[]

			for tags1 in all_tags:
				fix(tags1)
				index1=self.get_index([tags1], self.rev_tagset)[0]
				for z in range(len(index1)):
					for y in range(len(index1[z]), nl):
						index1[z].append(0)
				for z in range(len(index1), nl):
					index1.append(np.zeros(nl))

				all_index.append(index1)

				missing1=[]
				nll=0
				for idx, tag in enumerate(tags1):
					if idx > 0 and self.rev_tagset[tag].startswith("I-"):
						missing1.append(idx)
					else:
						nll+=1

				all_lens.append(nll)
				all_missing.append(missing1)

			all_index=torch.FloatTensor(np.array(all_index)).to(self.device)
			return all_tags, all_index, all_missing, all_lens

		all_tags1=all_tags2=all_tags3=None
		
		input_ids = input_ids.to(self.device)
		attention_mask = attention_mask.to(self.device)
		transforms = transforms.to(self.device)

		ll=lens.to(self.device)

		_, _, hidden_states = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
		if self.num_layers == 4:
			all_layers = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), 2)
		elif self.num_layers == 2:
			all_layers = torch.cat((hidden_states[-1], hidden_states[-2]), 2)

		# remove the opening [CLS]
		reduced=torch.matmul(transforms, all_layers)[:,1:,:] # reduced e' l'output di BERT

		if doEntities:

			# 3 LSTMs are used in cascade in addition to BERT!
			## LAYER 1
			lstm_out1, _ = self.lstm1(reduced)
			tag_space1 = self.hidden2tag1(lstm_out1)
			_, t1 = self.crf.viterbi_decode(tag_space1, ll-2)
			all_tags1, all_index1, all_missing1, n_lens1=get_layer_transformation(tag_space1, t1)
			input2=torch.matmul(all_index1,lstm_out1)

			## LAYER 2
			lstm_out2, _ = self.lstm2(input2)
			tag_space2 = self.hidden2tag2(lstm_out2)
			_, t2 = self.crf.viterbi_decode(tag_space2, torch.LongTensor(n_lens1) )
			all_tags2, all_index2, all_missing2, n_lens2=get_layer_transformation(tag_space2, t2)
			input3=torch.matmul(all_index2,lstm_out2)

			## LAYER 3
			lstm_out3, _ = self.lstm3(input3)
			tag_space3 = self.hidden2tag3(lstm_out3)
			_, t3 = self.crf.viterbi_decode(tag_space3, torch.LongTensor(n_lens2))
			all_tags3=[]
			for tags in t3:
				all_tags3.append(list(tags.data.cpu().numpy()))
			for tags3 in all_tags3:
				fix(tags3) # not going into details

			## Insert tokens into later layers that were compressed in earlier layers
			for idx, missing2 in enumerate(all_missing2):
				for m in missing2:
					parts=self.rev_tagset[all_tags3[idx][m-1]].split("-")
					if len(parts) > 1:
						all_tags3[idx].insert(m, self.tagset["I-%s" % parts[1]])
					else:
						all_tags3[idx].insert(m, self.tagset["O"])
			for idx, missing1 in enumerate(all_missing1):
				for m in missing1:
					parts=self.rev_tagset[all_tags3[idx][m-1]].split("-")
					if len(parts) > 1:
						all_tags3[idx].insert(m, self.tagset["I-%s" % parts[1]])
					else:
						all_tags3[idx].insert(m, self.tagset["O"])
			for idx, missing1 in enumerate(all_missing1):
				for m in missing1:
					parts=self.rev_tagset[all_tags2[idx][m-1]].split("-")
					if len(parts) > 1:
						all_tags2[idx].insert(m, self.tagset["I-%s" % parts[1]])
					else:
						all_tags2[idx].insert(m, self.tagset["O"])
			for idx in range(len(all_tags1)):
				all_tags2[idx]=all_tags2[idx][:len(all_tags1[idx])]
				all_tags3[idx]=all_tags3[idx][:len(all_tags1[idx])]

		return all_tags1, all_tags2, all_tags3

	def tag_all(self, batched_sents, batched_data, batched_mask, batched_transforms, batched_orig_token_lens, ordering, doEntities=True):
		
		""" Tag input data for layered sequence labeling """

		c=0
		ordered_preds=[]
		preds_in_order=None

		with torch.no_grad():
	
			for b in range(len(batched_data)):
				all_tags1, all_tags2, all_tags3 = self.predict_all(batched_data[b], attention_mask=batched_mask[b], transforms=batched_transforms[b], lens=batched_orig_token_lens[b], doEntities=doEntities)

				# for each sentence in the batch
				if doEntities:
					for d in range(len(all_tags1)):
						preds={}
						for entity in self.get_spans(self.rev_tagset, c, all_tags1[d], batched_orig_token_lens[b][d], batched_sents[b][d][1:]):
							preds[entity]=1
						for entity in self.get_spans(self.rev_tagset, c, all_tags2[d], batched_orig_token_lens[b][d], batched_sents[b][d][1:]):
							preds[entity]=1
						for entity in self.get_spans(self.rev_tagset, c, all_tags3[d], batched_orig_token_lens[b][d], batched_sents[b][d][1:]):
							preds[entity]=1
						ordered_preds.append(preds)
						c+=1
				
			if doEntities:
				preds_in_order = [None for i in range(len(ordering))]
				for i, ind in enumerate(ordering):
					preds_in_order[ind] = ordered_preds[i]

		return preds_in_order

	##################################################################
 	# utility functions
	def get_spans(self, rev_tagset, doc_idx, tags, length, sentence):
		
		# remove the opening [CLS] and closing [SEP]
		tags=tags[:length-2]
		
		entities={}

		for idx, tag in enumerate(tags):

			tag=rev_tagset[int(tag)]

			if tag.startswith("B-"):
				j=idx+1
				parts=tag.split("-")

				while(1):

					if j >= len(tags):
						break

					tagn=rev_tagset[int(tags[j])]
					if tagn.startswith("B") or tagn.startswith("O"):
						break

					parts_n=tagn.split("-")
		
					if parts_n[1] != parts[1]:
						break

					j+=1

				key=doc_idx, parts[1], idx, j

				entities[key]=1

		return entities

	def get_index(self, all_labels, rev_tagset):
		indices=[]
		for labels in all_labels:
			index=[]
			n=len(labels)
			for idx, label in enumerate(labels):
				ind=list(np.zeros(n))

				if label == -100 or not rev_tagset[label].startswith("I-"):
					ind[idx]=1		
					index.append(ind)
				else:
					index[-1][idx]=1
		
			indices.append(index)

		for index in indices:
			for i, idx in enumerate(index):
				idx=idx/np.sum(idx)

				index[i]=list(idx)

		return indices
	##################################################################
