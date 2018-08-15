# ACL2018_Multi_Input_OCR

This repository includes the implementation of the method from [our paper](http://www.ccs.neu.edu/home/dongrui/files/paper/acl_2018.pdf). It is implemented via tensorflow 1.9.0. Please use the following citation.

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

data_tokenize.py: create the vocabulary and tokenize the data.

flag.py: configuration of the model.

model.py: model construction.

model_attn.py: attntion model with different variations of attention combination strategies.

train.py: train the model.

decode.py: decode the model.




