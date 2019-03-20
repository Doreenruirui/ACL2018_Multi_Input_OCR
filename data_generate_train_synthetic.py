from os.path import join
import numpy as np
import argparse
from util import remove_nonascii


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='folder of data to be processed.')
    parser.add_argument('--prefix', type=str, help='prefix of the name of data file')
    parser.add_argument('--insertion', type=float, help='insertion error ratio')
    parser.add_argument('--deletion', type=float, help='delete error ratio')
    parser.add_argument('--substitution', type=float, help='substitution error ratio')
    args = parser.parse_args()
    return args


def get_train_single(folder_data, train, ins_ratio, del_ratio, sub_ratio):
    str_truth = ''              # get all the characters in file
    len_line = []               # get the number of characters in each line
    vocab = {}                  # get all the unique characters in file
    with open(join(folder_data, train + '.y.txt')) as f_:
        for line in f_:
            str_truth += line.strip()
            for ele in remove_nonascii(line.strip()):
                vocab[ele] = 1
            len_line.append(len(line.strip()))
    str_truth = list(str_truth)
    num_char = len(str_truth)
    print('Number of Characters in Corpus: %d' % num_char)
    vocab = vocab.keys()
    size_voc = len(vocab)
    print('Number of Unique Characters in Corpus: %d' % size_voc)
    error_ratio = ins_ratio + del_ratio + sub_ratio
    ins_v = ins_ratio / error_ratio
    del_v = (ins_ratio + del_ratio) / error_ratio
    num_error = int(np.floor(num_char * error_ratio))
    error_index = np.arange(num_char)
    np.random.shuffle(error_index)
    error_index = error_index[:num_error]                   # choose random positions to inject errors
    for char_id in error_index:
        rand_v = np.random.random()                         # choose an error type
        if 0 <= rand_v < ins_v:                                  # insertion error
            rand_index = np.random.choice(size_voc, 1)[0]   # choose an random character to insert
            str_truth[char_id] += vocab[rand_index]         # insert the character to the chosen position
        elif ins_v <= rand_v < del_v:                       # deletion error
            str_truth[char_id] = ''                         # delete the character from the chosen position
        else:                                               # substitution error
            cur_char = str_truth[char_id]                   # get the character to be substituted
            candidates = vocab[:]
            if cur_char in candidates:
                candidates.remove(cur_char)        # get the substitution candidates
            rand_index = np.random.choice(size_voc - 1, 1)[0]  # choose the substitution candidates
            str_truth[char_id] = candidates[rand_index]     # substitute the chosen character
    corrupted_lines = []
    start = 0
    with open(join(folder_data, train + '.x.txt'), 'w') as f_:  # write the corrupted string into lines
        for i in range(len(len_line)):
            corrupted_lines.append(''.join(str_truth[start: start + len_line[i]]))
            start += len_line[i]
            f_.write(corrupted_lines[i] + '\n')


def main():
    args = get_args()
    data_dir = args.data_dir
    prefix = args.prefix
    ins_ratio = args.insertion
    del_ratio = args.deletion
    sub_ratio = args.substitution
    get_train_single(data_dir, prefix, ins_ratio, del_ratio, sub_ratio)

if __name__ == '__main__':
    main()
