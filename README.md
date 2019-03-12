# ACL2018_Multi_Input_OCR

This repository includes the implementation of the method from [our paper](http://www.ccs.neu.edu/home/dongrui/paper/acl_2018.pdf). It is implemented via tensorflow 1.12.0. Please use the following citation.

@inproceedings{dong2018multi,
  title={Multi-Input Attention for Unsupervised OCR Correction},
  author={Dong, Rui and Smith, David},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={2363--2372},
  year={2018}
}

It contains the following file:

util.py: basic functions used by other scripts.

data_tokenize.py: 
	* create the vocabulary
		parameters:
			 data_dir: the directory where your training data is stored.
 			 prefix: the prefix of your file, e.g., "train", "dev", "test", here we set it as "train" to generate the vocabulary with the training files.
			 gen_voc: True for generating a new vocabulary file, Flase for tokenizing the given files with a existing vocabulary, here we set it as True
		INPUT: It takes "DATA_DIR/train.x.txt" and "DATA_DIR/train.y.txt" for creating the vocabulary, each line in "train.x.txt" from OCR ou/tput, and each line in "train.y.txt" is the manually transcribed target for the corresponding line in "train.x.txt"
		Output file: "DATA_DIR/vocab.dat" the vocabulary file where each line is a character.
	* tokenize a given file:	
		parameters:	
			 data_dir: the directory where the file your want to tokenize is stored		 
			 voc_dir: the directory where your vocabulary file is stored	   
 			 prefix: the prefix of your files to be tokenized, e.g., "train", "dev", "test", here we set it as "train" to generate the vocabulary.
			 gen_voc: set it as False for tokenzing the given files with a exisiting vocabulary file
		It takes "DATA_DIR/PREFIX.x.txt" and "DATA_DIR/PREFIX.y.txt" for tokenize your file with vocabulary file

flag.py: configuration of the model.

model.py: model construction.

model_attn.py: attntion model with different variations of attention combination strategies.

train.py: train the model.

decode.py: decode the model.

run_train.sh: script to train a basic correction model

run_decode.sh: script to decode a model

The input file to train.py should be:
	train.ids.x, train.ids.y: the  


