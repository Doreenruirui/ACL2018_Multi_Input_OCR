import os
from os.path import join as pjoin
import util
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp', help='Data directory')
    parser.add_argument('--voc_dir', type=str, default='/tmp', help='Data directory')
    parser.add_argument('--gen_voc', type=lambda x: x.lower() == 'true', default=False,
                        help='True/False: whether to create a vocabulary from the input file.')
    parser.add_argument('--flag_ascii', type=lambda x: x.lower() == 'true', default=False,
                        help='True/False: whether to create a vocabulary with ascii characters.')
    parser.add_argument('--prefix', type=str, default=None,
                        help='train/dev/test: prefix of the file to be tokenzied.')
    args = parser.parse_args()
    return args


def create_vocabulary(data_dir, voc_dir, file_prefix, flag_ascii=False):
    def tokenize_file(path):
        print "Create vocabulary from file: %s" % path
        for line in file(path):
            line = line.strip()
            if flag_ascii:
                line = util.remove_nonascii(line)
            tokens = list(line)
            for ch in tokens:
                vocab[ch] = vocab.get(ch, 0) + 1
    path_x = pjoin(data_dir, file_prefix + '.x.txt')
    path_y = pjoin(data_dir, file_prefix + '.y.txt')
    path_vocab = os.path.join(data_dir, "vocab.dat")
    print "Vocabulary file: %s" % path_vocab
    vocab = {}
    tokenize_file(path_x)
    tokenize_file(path_y)
    vocab_list = util._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Vocabulary size: %d" % len(vocab_list))
    with open(path_vocab, mode="wb") as f:
        for ch in vocab_list:
            f.write(ch + "\n")

def data_to_token_ids(data_path, target_path, vocab, flag_ascii=False):
    print("Tokenizing data in %s" % data_path)
    with open(data_path, mode="r") as data_file:
        with open(target_path, mode="w") as tokens_file:
            for line in data_file:
                line = line.strip('\n')
                token_ids = util.sentenc_to_token_ids(line, vocab, flag_ascii)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def tokenize_data(data_dir, prefix, flag_ascii=False):
    path_vocab = os.path.join(data_dir, "vocab.dat")
    vocab, _  = util.read_vocab(path_vocab)
    path_x = os.path.join(data_dir,  prefix + ".x.txt")
    path_y = os.path.join(data_dir, prefix + ".y.txt")
    y_ids_path = os.path.join(data_dir, prefix + ".ids.y")
    x_ids_path = os.path.join(data_dir, prefix + ".ids.x")
    data_to_token_ids(path_x, x_ids_path, vocab)
    data_to_token_ids(path_y, y_ids_path, vocab)
    return (x_ids_path, y_ids_path)


def main():
    args = get_args()
    if args.gen_voc:
        create_vocabulary(args.data_dir, args.voc_dir, args.prefix, flag_ascii=args.flag_ascii)
    if args.prefix is not None:
        tokenize_data(args.data_dir, args.prefix, flag_ascii=args.flag_ascii)

if __name__ == "__main__":
    main()