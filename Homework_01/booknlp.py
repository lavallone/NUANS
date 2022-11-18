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

			tagsetPath="labels/LitBank_entity_categories.txt" # questo sara' da modificare per farlo funzionare con solo {PER,LOC,ORG}
			tagsetPath = pkg_resources.resource_filename(__name__, tagsetPath)

			if self.doEntities:
				self.entityTagger = LitBankEntityTagger(self.entityPath, tagsetPath) # WRAPPER of the model which performs NER tag

			self.tagger = spacy_tokenizer.SpacyPipeline(spacy_nlp) # we exploit the capabilities of Spacy
			print("--- startup: %.3f seconds ---" % (time.time() - start_time))
			
	def process(self, filename, outFolder, name): # questa funzione viene chiamata durante l'esecuzione della pipeline	

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
				
				d = {"tokens" : [], "entities" : []}
				f = open(outFolder+name+".json", "w")
    
				tokens = self.tagger.tag(data) # it returns a list of tokens of the class 'pipelines.Token' --> which have already some informations attached (like POS tag) thanks to Spacy
				for token in tokens:
					token_dict = {"word" : token.text, "start_offset" : token.startByte, "end_offset" : token.endByte, "POS" : token.pos}	
					d["tokens"].append(token_dict)
     
    			# with open(join(outFolder, "%s_tokens.tsv" % (name)), "w", encoding="utf-8") as out:
				# 		out.write("%s\n" % '\t'.join(["paragraph_ID", "sentence_ID", "token_ID_within_sentence", "token_ID_within_document", "word", "lemma", "byte_onset", "byte_offset", "POS_tag", "fine_POS_tag", "dependency_relation", "syntactic_head_ID", "event"]))
				# 		for token in tokens:
				# 			out.write("%s\n" % token)
       
				print("--- spacy: %.3f seconds ---" % (time.time() - start_time))

				if self.doEntities:
					start_time = time.time()
					# we give the processed tokens by Spacy to the model for tagging them
					entity_vals = self.entityTagger.tag(tokens, doEntities=self.doEntities) # here we perform the predictions
					entity_vals["entities"] = sorted(entity_vals["entities"]) # e' una lista di named entities rappresentate in questo modo --> (start_token, phraseEndToken, label, phrase)

					print("--- entities: %.3f seconds ---" % (time.time() - start_time))
					start_time=time.time()

					entities = entity_vals["entities"]
					
					for start, end, cat, text in entities:
						ner_prop=cat.split("_")[0]
						ner_type=cat.split("_")[1]
						entity_dict = {"text" : text, "start_token" : start, "end_token" : end, "ner_prop" : ner_prop, "ner" : ner_type}
						d["entities"].append(entity_dict)
      
     				# with open(join(outFolder, "%s_entities.tsv" % (name)), "w", encoding="utf-8") as out:
					# 	out.write("start_token\tend_token\tprop\tcat\ttext\n")
					# 	for start, end, cat, text in entities:
					# 		ner_prop=cat.split("_")[0]
					# 		ner_type=cat.split("_")[1]
					# 		out.write("%s\t%s\t%s\t%s\t%s\n" % (start, end, ner_prop, ner_type, text))

					json.dump(d, f)
					f.close()

				print("--- TOTAL (excl. startup): %.3f seconds ---, %s words" % (time.time() - originalTime, len(tokens)))
				return time.time() - originalTime