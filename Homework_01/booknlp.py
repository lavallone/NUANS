import spacy
from utilities import spacy_tokenizer
from entity_tagger import LitBankEntityTagger
from os.path import join
import os
import json
from collections import Counter
import time
from pathlib import Path
import urllib.request 
import pkg_resources
import torch

class BookNLP:

	def __init__(self, model_params):

		with torch.no_grad():
			start_time = time.time()
			print(model_params)
   
			# we use spacy only for dependency tree and POS TAGGING (this last one will be used for our NER task)
			spacy_model="en_core_web_sm"

			spacy_nlp = spacy.load(spacy_model, disable=["ner"]) # we disable the ner task for spacy!

			pipes=model_params["pipeline"].split(",") # it'll always be pipes="entity"

			###########################################################################################################################
			## MODEL loading part
			home = str(Path.home())
			modelPath = os.path.join(home, "booknlp_models")
			if not Path(modelPath).is_dir(): # in case I create the folder "booknlp_models"
				Path(modelPath).mkdir(parents=True, exist_ok=True)

			if model_params["model"] == "very big": # if I want to use the biggest model (459 M)
				entityName="entities_google_bert_uncased_L-12_H-768_A-12.model"
				self.entityPath=os.path.join(modelPath, entityName)
				if not Path(self.entityPath).is_file():
					print("downloading %s" % entityName)
					urllib.request.urlretrieve("http://ischool.berkeley.edu/~dbamman/booknlp_models/%s" % entityName, self.entityPath)
			if model_params["model"] == "big": # if I want to use the big model (297 M)
				entityName="entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"
				self.entityPath=os.path.join(modelPath, entityName)
				if not Path(self.entityPath).is_file():
					print("downloading %s" % entityName)
					urllib.request.urlretrieve("http://ischool.berkeley.edu/~dbamman/booknlp_models/%s" % entityName, self.entityPath)
			elif model_params["model"] == "medium": # if I want to use the medium model (137 M)
				entityName="entities_google_bert_uncased_L-4_H-512_A-8.model"
				self.entityPath=os.path.join(modelPath, entityName)
				if not Path(self.entityPath).is_file():
					print("downloading %s" % entityName)
					urllib.request.urlretrieve("http://ischool.berkeley.edu/~dbamman/booknlp_models/%s" % entityName, self.entityPath)
			elif model_params["model"] == "small": # if I want to use the little one (57 M)
				entityName="entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model"
				self.entityPath=os.path.join(modelPath, entityName)
				if not Path(self.entityPath).is_file():
					print("downloading %s" % entityName)
					urllib.request.urlretrieve("http://ischool.berkeley.edu/~dbamman/booknlp_models/%s" % entityName, self.entityPath)
			###########################################################################################################################

			self.doEntities=False

			for pipe in pipes:
				if pipe == "entity":
					self.doEntities=True

			tagsetPath="LitBank_entity_categories.txt" # questo sara' da modificare per farlo funzionare con solo {PER,LOC,ORG}
			tagsetPath = pkg_resources.resource_filename(__name__, tagsetPath)

			if self.doEntities:
				self.entityTagger = LitBankEntityTagger(self.entityPath, tagsetPath) # WRAPPER of the model which performs NER tag

			self.tagger = spacy_tokenizer.SpacyPipeline(spacy_nlp) # we exploit the capabilities of Spacy
			print("--- startup: %.3f seconds ---" % (time.time() - start_time))

	# da vedere se cancellare la funzione
	def get_syntax(self, tokens, entities, assignments, genders):

		def check_conj(tok, tokens):
			if tok.deprel == "conj" and tok.dephead != tok.token_id:
				# print("found conj", tok.text)
				return tokens[tok.dephead]
			return tok

		def get_head_in_range(start, end, tokens):
			for i in range(start, end+1):
				if tokens[i].dephead < start or tokens[i].dephead > end:
					return tokens[i]
			return None

		agents={}
		patients={}
		poss={}
		mods={}
		prop_mentions={}
		pron_mentions={}
		nom_mentions={}
		keys=Counter()


		toks_by_children={}
		for tok in tokens:
			if tok.dephead not in toks_by_children:
				toks_by_children[tok.dephead]={}
			toks_by_children[tok.dephead][tok]=1

		for idx, (start_token, end_token, cat, phrase) in enumerate(entities):
			ner_prop=cat.split("_")[0]
			ner_type=cat.split("_")[1]

			if ner_type != "PER":
				continue

			coref=assignments[idx]

			keys[coref]+=1
			if coref not in agents:
				agents[coref]=[]
				patients[coref]=[]
				poss[coref]=[]
				mods[coref]=[]
				prop_mentions[coref]=Counter()
				pron_mentions[coref]=Counter()
				nom_mentions[coref]=Counter()

			if ner_prop == "PROP":
				prop_mentions[coref][phrase]+=1
			elif ner_prop == "PRON":
				pron_mentions[coref][phrase]+=1
			elif ner_prop == "NOM":
				nom_mentions[coref][phrase]+=1


			tok=get_head_in_range(start_token, end_token, tokens)
			if tok is not None:

				tok=check_conj(tok, tokens)
				head=tokens[tok.dephead]

				# nsubj
				# mod
				if tok.deprel == "nsubj" and head.lemma == "be":
					for sibling in toks_by_children[head.token_id]:

						# "he was strong and happy", where happy -> conj -> strong -> attr/acomp -> be
						sibling_id=sibling.token_id
						sibling_tok=tokens[sibling_id]
						if (sibling_tok.deprel == "attr" or sibling_tok.deprel == "acomp") and (sibling_tok.pos == "NOUN" or sibling_tok.pos == "ADJ"):
							mods[coref].append({"w":sibling_tok.text, "i":sibling_tok.token_id})

							if sibling.token_id in toks_by_children:
								for grandsibling in toks_by_children[sibling.token_id]:
									grandsibling_id=grandsibling.token_id
									grandsibling_tok=tokens[grandsibling_id]

									if grandsibling_tok.deprel == "conj" and (grandsibling_tok.pos == "NOUN" or grandsibling_tok.pos == "ADJ"):
										mods[coref].append({"w":grandsibling_tok.text, "i":grandsibling_tok.token_id})



				# ("Bill and Ted ran" conj captured by check_conj above)
				elif tok.deprel == "nsubj" and head.pos == ("VERB"):
					agents[coref].append({"w":head.text, "i":head.token_id})

				# "Bill ducked and ran", where ran -> conj -> ducked
					for sibling in toks_by_children[head.token_id]:
						sibling_id=sibling.token_id
						sibling_tok=tokens[sibling_id]
						if sibling_tok.deprel == "conj" and sibling_tok.pos == "VERB":
							agents[coref].append({"w":sibling_tok.text, "i":sibling_tok.token_id})
				
				# "Jack was hit by John and William" conj captured by check_conj above
				elif tok.deprel == "pobj" and head.deprel == "agent":
					# not root
					if head.dephead != head.token_id:
						grandparent=tokens[head.dephead]
						if grandparent.pos.startswith("V"):
							agents[coref].append({"w":grandparent.text, "i":grandparent.token_id})


				# patient ("He loved Bill and Ted" conj captured by check_conj above)
				elif (tok.deprel == "dobj" or tok.deprel == "nsubjpass") and head.pos == "VERB":
					patients[coref].append({"w":head.text, "i":head.token_id})


				# poss

				elif tok.deprel == "poss":
					poss[coref].append({"w":head.text, "i":head.token_id})

					# "her house and car", where car -> conj -> house
					for sibling in toks_by_children[head.token_id]:
						sibling_id=sibling.token_id
						sibling_tok=tokens[sibling_id]
						if sibling_tok.deprel == "conj":
							poss[coref].append({"w":sibling_tok.text, "i":sibling_tok.token_id})
					

		data={}
		data["characters"]=[]

		for coref, total_count in keys.most_common():

			# must observe a character at least *twice*

			if total_count > 1:
				chardata={}
				chardata["agent"]=agents[coref]
				chardata["patient"]=patients[coref]
				chardata["mod"]=mods[coref]
				chardata["poss"]=poss[coref]
				chardata["id"]=coref
				if coref in genders:
					chardata["g"]=genders[coref]
				else:
					chardata["g"]=None
				chardata["count"]=total_count

				mentions={}

				pnames=[]
				for k,v in prop_mentions[coref].most_common():
					pnames.append({"c":v, "n":k})
				mentions["proper"]=pnames

				nnames=[]
				for k,v in nom_mentions[coref].most_common():
					nnames.append({"c":v, "n":k})
				mentions["common"]=nnames

				prnames=[]
				for k,v in pron_mentions[coref].most_common():
					prnames.append({"c":v, "n":k})
				mentions["pronoun"]=prnames

				chardata["mentions"]=mentions

				
				data["characters"].append(chardata)
			
		return data
			
	def process(self, filename, outFolder, idd): # questa funzione viene chiamata durante l'esecuzione della pipeline	

		with torch.no_grad():

			start_time = time.time()
			originalTime=start_time

			with open(filename) as file:
				data = file.read() # single story to process
				if len(data) == 0:
					print("Input file is empty: %s" % filename)
					return 
				try:
					os.makedirs(outFolder)
				except FileExistsError:
					pass
					
				tokens = self.tagger.tag(data) # it returns a list of tokens of the class 'pipelines.Token' --> which have already some informations attached (like POS tag) thanks to Spacy
				with open(join(outFolder, "%s_tokens.tsv" % (idd)), "w", encoding="utf-8") as out:
						out.write("%s\n" % '\t'.join(["paragraph_ID", "sentence_ID", "token_ID_within_sentence", "token_ID_within_document", "word", "lemma", "byte_onset", "byte_offset", "POS_tag", "fine_POS_tag", "dependency_relation", "syntactic_head_ID", "event"]))
						for token in tokens:
							out.write("%s\n" % token)
				print("--- spacy: %.3f seconds ---" % (time.time() - start_time))

				if self.doEntities:
					start_time = time.time()
					# we give the processed tokens by Spacy to the model for tagging them
					entity_vals = self.entityTagger.tag(tokens, doEntities=self.doEntities) # here we perform the predictions
					entity_vals["entities"] = sorted(entity_vals["entities"]) # e' una lista di named entities rappresentate in questo modo --> (start_token, phraseEndToken, label, phrase)

					print("--- entities: %.3f seconds ---" % (time.time() - start_time))
					start_time=time.time()

					entities = entity_vals["entities"]
					with open(join(outFolder, "%s_entities.tsv" % (idd)), "w", encoding="utf-8") as out:
						out.write("start_token\tend_token\tprop\tcat\ttext\n")
						for start, end, cat, text in entities:
							ner_prop=cat.split("_")[0]
							ner_type=cat.split("_")[1]
							out.write("%s\t%s\t%s\t%s\t%s\n" % (start, end, ner_prop, ner_type, text))

				print("--- TOTAL (excl. startup): %.3f seconds ---, %s words" % (time.time() - originalTime, len(tokens)))
				return time.time() - originalTime