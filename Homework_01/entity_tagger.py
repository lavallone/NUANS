from tagger import Tagger
import torch
import re
from utilities import layered_reader
from utilities import sequence_layered_reader
import pkg_resources

class LitBankEntityTagger:
	def __init__(self, model_file, model_tagset):

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tagset = sequence_layered_reader.read_tagset(model_tagset) # fa una semplice lettura dei possibili tag per eseguire la NER
		
  		# the model also tags 
		supersenseTagset = pkg_resources.resource_filename(__name__, "supersense.tagset")
		self.supersense_tagset=sequence_layered_reader.read_tagset(supersenseTagset)

		############################################################################################################################################################################
		## WE LOAD THE ACTUAL MODEL WHICH PERFORMS NER TAGGING
		base_model=re.sub("google_bert", "google/bert", model_file.split("/")[-1])
		base_model=re.sub(".model", "", base_model)
		self.model = Tagger(freeze_bert=False, base_model=base_model, tagset_flat={"EVENT":1, "O":1}, supersense_tagset=self.supersense_tagset, tagset=self.tagset, device=device)
		self.model.to(device)
		self.model.load_state_dict(torch.load(model_file, map_location=device)) # we load the pretrained weights
		############################################################################################################################################################################
  
		wnsFile = pkg_resources.resource_filename(__name__, "wordnet.first.sense") # non so a cosa possa servire
		self.wns = self.read_wn(wnsFile)

	def read_wn(self, filename):
		wns={}
		with open(filename) as file:
			for line in file:
				cols=line.rstrip().split("\t")
				word=cols[0]
				pos=cols[1]
				wn=int(cols[2].split(" ")[0])
				wns["%s.%s" % (word, pos)]=wn
		return wns

	def get_wn(self, supersense_batched_sents):

		wn_batches=[]

		for idx, b_sent in enumerate(supersense_batched_sents):

			max_len=0
			for sent in b_sent:
				if sent is not None:
					if len(sent) > max_len:
						max_len=len(sent)

			wn_senses=[]

			for sent in b_sent:

				wn=[]
				if sent is None:
					continue

				for word in sent:

					if word is None:
						wn.append(0)
					else:

						text=word.text
						pos=word.pos
						if pos == "NOUN":
							pos="n"
						elif pos == "VERB":
							pos="v"
						term=text.split(" ")[-1].lower()
						key="%s.%s" % (term, pos)
						if key in self.wns:
							wn.append(self.wns[key])
						else:
							wn.append(1)

				for val in range(len(sent), max_len):
					wn.append(0)
				wn_senses.append(wn)

			wn_senses=torch.LongTensor(wn_senses)
			wn_batches.append(wn_senses)
		return wn_batches

	def tag(self, toks, doEntities=True): # toks are the tokens of the entire text

		max_sentence_length=500
		entities=[]

		batch_size=32
		sents=[]
		o_sents=[]
		sent=[]
		o_sent=[]
		lastSid=None

		length=0
		for tok in toks:
			wptok=tok.text
			# working with uncased BERT models, so add a special tag to denote capitalization --> ecco a che serve!
			if wptok[0].lower() != wptok[0]: # se la prima lettera e' una maiuscola
				wptok="[CAP] " + wptok.lower()

			toks=self.model.tokenizer.tokenize(wptok) # tokenizziamo il 'token' per darlo in input a BERT!
			if lastSid is not None and ( tok.sentence_id != lastSid or length+len(toks)>max_sentence_length ):
				sents.append(sent)
				o_sents.append(o_sent)
				sent=[]
				o_sent=[]
				length=0
			
			sent.append(toks)
			o_sent.append(tok)

			lastSid=tok.sentence_id
			length+=len(toks)
		sents.append(sent)
		o_sents.append(o_sent)

		sentences=[]
		o_sentences=[]

		max_sentence_length=500
		sentence=[ ["[CLS]"] ]
		o_sent=[]
		cur_length=0 # current length

		for idx, sent in enumerate(sents):

			sent_len=0
			for toks in sent:
				sent_len+=len(toks)

			if sent_len + cur_length >= max_sentence_length:
				sentence.append(["[SEP]"])
				sentences.append(sentence)
				o_sentences.append(o_sent)
				sentence=[["[CLS]"]]
				o_sent=[]
				cur_length=0

			cur_length+=sent_len

			sentence.extend(sent)
			o_sent.extend(o_sents[idx])


		if len(sentence) > 1:		
			sentence.append(["[SEP]"])
			o_sentences.append(o_sent)
			sentences.append(sentence)

		sents=o_sentences

		# da tokens a 'batch numerici'
		batched_sents, batched_data, batched_mask, batched_transforms, batched_orig_token_lens, ordering, order_to_batch_map = layered_reader.get_batches(self.model, sentences, batch_size, self.tagset, training=False)

		batch_pos={}
		for idx, ind in enumerate(ordering):
			batch_id, batch_s, batch_position=order_to_batch_map[idx]
			if batch_id not in batch_pos:
				batch_pos[batch_id]=[None]*batch_s
			batch_pos[batch_id][batch_position]=[None]
			for tok in sents[ind]:
				# print(tok)
				batch_pos[batch_id][batch_position].append(tok)
			batch_pos[batch_id][batch_position].append(None)

		batched_pos=[None]*len(batch_pos)
		for i in range(len(batch_pos)):
			batched_pos[i]=batch_pos[i]


		# qui avvengono le predizioni delle named entities
		preds_in_order = self.model.tag_all(batched_sents, batched_data, batched_mask, batched_transforms, batched_orig_token_lens, ordering, doEntities=doEntities)
		
		return_vals={}

		if doEntities:
			for idx, preds in enumerate(preds_in_order):
				for _, label, start, end in preds:
					start_token=sents[idx][start].token_id
					end_token=sents[idx][end-1].token_id
					phrase=' '.join([x.text for x in sents[idx][start:end]])
					phraseEndToken=int(end_token)
					if phraseEndToken == -2:
						phraseEndToken=start_token
					entities.append((start_token, phraseEndToken, label, phrase))
			return_vals["entities"]=entities

		return return_vals