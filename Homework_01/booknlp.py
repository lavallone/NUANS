import sys
import argparse
from transformers import logging
logging.set_verbosity_error()
from pathlib import Path
import os
from english_booknlp import EnglishBookNLP

class BookNLP():

	def __init__(self, model_params):
		self.booknlp=EnglishBookNLP(model_params)

	def process(self, inputFile, outputFolder, idd):
		self.booknlp.process(inputFile, outputFolder, idd)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--inputFile', help='Filename to run BookNLP on', required=True)
	parser.add_argument('-o','--outputFolder', help='Folder to write results to', required=True)
	parser.add_argument('--id', help='ID of text (for creating filenames within output folder)', required=True)

	args = vars(parser.parse_args())

	inputFile=args["inputFile"]
	outputFolder=args["outputFolder"]
	idd=args["id"]

	print("tagging %s" % inputFile)

	model_params={"pipeline":"entity", "model":"big",}

	booknlp=BookNLP(model_params)
	booknlp.process(inputFile, outputFolder, idd)