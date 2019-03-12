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

#### util.py: basic functions used by other scripts.

#### data_tokenize.py: 
* create the vocabulary
    * Parameters:
        * data_dir: the directory where your training data is stored.
        * prefix: the prefix of your file, e.g., **train**, **dev**, **test**, here we set it as **train** to generate the vocabulary with the training files.
	    * gen_voc: **True** for generating a new vocabulary file, **False** for tokenizing the given files with a existing vocabulary, here we set it as **True**.
	* INPUT: It takes **DATA_DIR/train.x.txt** and **DATA_DIR/train.y.txt** for creating the vocabulary, each line in "train.x.txt" from OCR ou/tput, and each line in "train.y.txt" is the manually transcribed target for the corresponding line in "train.x.txt"
	* OUTPUT: **DATA_DIR/vocab.dat** the vocabulary file where each line is a character.
* tokenize a given file:	
    * Parameters:	
	    * data_dir: the directory where the file your want to tokenize is stored		 
		* voc_dir: the directory where your vocabulary file is stored	   
 		* prefix: the prefix of your files to be tokenized, e.g., "train", "dev", "test", here we set it as "train" to generate the vocabulary.
		* gen_voc: set it as False for tokenzing the given files with a exisiting vocabulary file
	* INPUT: It takes **DATA_DIR/PREFIX.x.txt** and **DATA_DIR/PREFIX.y.txt** as input and tokenize them with the given vocabulary file
	* OUTPUT: **DATA_DIR/PREFIX.ids.x** and **DATA_DIR/PREFIX.ids.y**, the tokenized files where each line is the id of each character for the line in the corresponding input file

#### flag.py: configuration of the model.

#### model.py: construct the correction model 
   It is an attention-based seq2seq model modified based on the [neural language correction](https://github.com/stanfordmlgroup/nlc) model.

#### model_attn.py: attention model with different attention combination strategies: "single", "average", "weight", "flat"

#### train.py: train the model.
* Basic Parameters:
    * data_dir: the directory of training and development files
    * voc_dir: the directory of the vocabulary file
    * train_dir: the directory to store the trained model
    * num_layers: number of layers of LSTM
    * size: the hidden size of LSTM unit
* INPUT: It takes the tokenized training files **DATA_DIR/train.ids,x**, **DATA_DIR/train.ids.y** and development files **DATA_DIR/dev.ids.x**, **DATA_DIR/dev.y.ids** as well as the vocabulary file as input, train the model on the training files and evaluate the model on the development files to decide whether to store a new checkpoint.
* OUTPUT: A correction model.

#### decode.py: decode the model.
* Basic Parameters:
    * data_dir: the directory of test file
    * voc_dir: the directory of the vocabulary file
    * train_dir: the directory where the trained model is stored
    * out_dir: the directory to store the output files
    * num_layers: number of layers of LSTM
    * size: the hidden size of LSTM unit
    * decode: the decoding strategy to use: **single**, **average**, **weight**, **flat**
    * beam_size:  beam search width
* INPUT: It takes the test files **DATA_DIR/test.x.txt**, **DATA_DIR/test.y.txt**, the vocabulary file **VOC_DIR/vocab.dat**, and the trained model  **TRAIN_DIR/best-ckpt** as input for decoding.
* OUTPUT: It output the decoding results in two files:
  * **OUT_DIR/test.DECODE.o.txt**: storing the top **BEAM_SIZE** decoding suggestions for each line in test file. It has **N * BEAM_SIZE** lines, every line in **test.x.txt** corresponds to **BEAM_SIZE** lines in this file.
  * **OUT_DIR/test.DECODE.p.txt**: storing the probability of each decoding suggestion in the above file.



