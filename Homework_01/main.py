import sys
import argparse
from transformers import logging
logging.set_verbosity_error()
from pathlib import Path
import os
from booknlp import BookNLP

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--inputFile', help='Filename to run BookNLP on', required=True)
	parser.add_argument('-o','--outputFolder', help='Folder to write results to', required=True)
	parser.add_argument('--id', help='ID of text (for creating filenames within output folder)', required=True)
	parser.add_argument('--model_size', help='Size of the pretrained model', required=True)

	args = vars(parser.parse_args())

	inputFile=args["inputFile"]
	outputFolder=args["outputFolder"]
	idd=args["id"]
	model_size=args["model_size"]

	print("tagging %s" % inputFile)

	model_params={"pipeline":"entity", "model":model_size,} # we exploit only the named entity tagger of the BookNLP pipeline!
	booknlp = BookNLP(model_params)
	booknlp.process(inputFile, outputFolder, idd)