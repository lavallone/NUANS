import argparse
from transformers import logging
logging.set_verbosity_error()
import os
from tqdm import tqdm
from booknlp import BookNLP

def extract_name(s):
	pos=0
	for i,c in enumerate(s):
		if c == "/":
			pos=i
	s = s[pos+1:]
	for i,c in enumerate(s[::-1]):
		if c == ".":
			pos = i
			break
	return s[:-pos-1]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFairyFolder', help='Folder containing Fairy Tales', required=True)
	parser.add_argument('--inputShortFolder', help='Folder containing Short Stories', required=True)
	#parser.add_argument('-i','--inputFile', help='Filename to run BookNLP on', required=True)
	#parser.add_argument('-o','--outputFolder', help='Folder to write results to', required=True)
	#parser.add_argument('--id', help='ID of text (for creating filenames within output folder)', required=True)
	parser.add_argument('--model_size', help='Size of the pretrained model', required=True)

	args = vars(parser.parse_args())

	#stories_list = args["stories_list"]
	inputFairyFolder = args["inputFairyFolder"]
	inputShortFolder = args["inputShortFolder"]
	fairy_tales_list = [inputFairyFolder+s for s in os.listdir(inputFairyFolder)]
	short_stories_list = [inputShortFolder+s for s in os.listdir(inputShortFolder)]
	stories_list = fairy_tales_list + short_stories_list
	print(stories_list)
	#inputFile=args["inputFile"]
	#outputFolder=args["outputFolder"]
	outputFolder="/content/results/" # here I'm going to save the computed named entities!
	#idd=args["id"]
	model_size=args["model_size"]
 
	# model_params={"pipeline":"entity", "model":model_size,} # we exploit only the named entity tagger of the BookNLP pipeline!
	# booknlp = BookNLP(model_params)

	# for s in tqdm(stories_list):
	# 	inputFile = s
	# 	name = extract_name(s)
	# 	print("tagging %s" % inputFile)
	# 	booknlp.process(inputFile, outputFolder, name)