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
	parser.add_argument('--process_all', help='Selection of stories to process', required=False)
	parser.add_argument('--model_size', help='Size of the pretrained model', required=True)

	args = vars(parser.parse_args())
	outputFolder="/content/results/" # here I'm going to save the computed named entities!
	model_size=args["model_size"]
 
	model_params={"pipeline":"entity", "model":model_size,} # we exploit only the named entity tagger of the BookNLP pipeline!
	booknlp = BookNLP(model_params)

	if args["process_all"] == True:
		inputFairyFolder = args["inputFairyFolder"]
		inputShortFolder = args["inputShortFolder"]
		fairy_tales_list = [inputFairyFolder+s for s in os.listdir(inputFairyFolder)]
		short_stories_list = [inputShortFolder+s for s in os.listdir(inputShortFolder)]
		stories_list = fairy_tales_list + short_stories_list
	else:
		stories_list = ["/content/FAIRY_TALE/texts/bn_14140242n_The Frog Prince.txt", \
               "/content/FAIRY_TALE/texts/bn_03393628n_Cannetella.txt", \
               "/content/FAIRY_TALE/texts/bn_03884049n_The Enchanted Snake.txt", \
               "/content/FAIRY_TALE/texts/bn_01899260n_The Raven.txt", \
               "/content/SHORT_STORY/texts/bn_00955099n_The Nameless City.txt", \
               "/content/SHORT_STORY/texts/bn_03149445n_The Tree.txt", \
               "/content/SHORT_STORY/texts/bn_03297228n_.007.txt", \
               "/content/SHORT_STORY/texts/bn_03419493n_A Country Doctor.txt"]
     
	for s in tqdm(stories_list):
		inputFile = s
		name = extract_name(s)
		print("tagging %s" % inputFile)
		booknlp.process(inputFile, outputFolder, name)