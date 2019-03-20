from os.path import join, exists
import numpy as np
import os
import kenlm
from os.path import join as pjoin
from multiprocessing import Pool
from util import remove_nonascii
import argparse


folder_multi = '/gss_gpfs_scratch/dong.r/Dataset/OCR/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='folder of data.')
    parser.add_argument('--lm_file', type=str, help='trained language model file.')
    parser.add_argument('--out_dir', type=str, help='folder of output.')
    parser.add_argument('--prefix',  type=str, help='train/test/dev: prefix of the file name.')
    parser.add_argument('--flag_manual', type=lambda x: x.lower() == 'true',
                        help='True/False: whether the input file has corresponding manual transcription.')
    parser.add_argument('--lm_prob', type=float,
                        help='the threshold of the language model score to filter too noisy data.')
    parser.add_argument('--start', type=int, help='the start line no of the test file to process.')
    parser.add_argument('--end', type=int, help='the ending line no of the test file to process.')
    args = parser.parse_args()
    return args


def initialize(file_lm):
    global lm
    lm = kenlm.LanguageModel(file_lm)


def get_string_to_score(sent):
    sent = remove_nonascii(sent)
    items = []
    for ele in sent:
        if len(ele.strip()) == 0:
            items.append('<space>')
        else:
            items.append(ele)
    return ' '.join(items)


def score_sent(paras):
    global lm
    thread_no, sent = paras
    sent = get_string_to_score(sent.lower())
    return thread_no, lm.perplexity(sent)


def rank_sent(pool, sents):    # find the best sentence with lowest perplexity
    sents = [ele.lower() for ele in sents]
    probs = np.ones(len(sents)) * -1
    results = pool.map(score_sent, zip(np.arange(len(sents)), sents))
    min_str = ''
    min_prob = float('inf')
    min_id = -1
    for tid, score in results:
        cur_prob = score
        probs[tid] = cur_prob
        if cur_prob < min_prob:
            min_prob = cur_prob
            min_str = sents[tid]
            min_id = tid
    return min_str, min_id, min_prob, probs


def generate_train_noisy(data_dir, out_dir, file_prefix, lm_file, lm_score, flag_manual, start, end):
    def read_file(path):
        line_id = 0
        res = []
        with open(path) as f_:
            for line in f_:
                if line_id >= start:
                    res.append(line)
                    if line_id + 1 == end:
                        break
                line_id += 1
        return res
    list_info = read_file(join(data_dir, file_prefix + '.info.txt'))
    list_x = read_file(join(data_dir, file_prefix + '.x.txt'))
    if flag_manual:            # if current OCR'd file has corresponding manual transcription
        list_y = read_file(join(data_dir, file_prefix + '.y.txt'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    f_x = open(join(out_dir, '%s.x.txt.%d_%d'%(file_prefix, start, end)), 'w')
    f_y = open(join(out_dir, '%s.y.txt.%d_%d'%(file_prefix, start, end)), 'w')
    f_info = open(join(out_dir, '%s.info.txt.%d_%d'%(file_prefix, start, end)), 'w')
    if flag_manual:
        f_z = open(join(out_dir, '%s.z.txt.%d_%d'%(file_prefix, start, end)), 'w')
    pool = Pool(100, initializer=initialize(lm_file))
    for i in range(len(list_x)):
        witness = [ele.strip() for ele in list_x[i].strip('\n').split('\t') if len(ele.strip()) > 0]
        best_str, best_id, best_prob, probs = rank_sent(pool, witness)
        if best_prob < 10 and best_prob < probs[0]:
            if probs[0] - best_prob > 1:
                f_x.write(witness[0] + '\n')
                f_y.write(best_str + '\n')
                f_info.write(list_info[i])
                if flag_manual:
                    f_z.write(list_y[i])
    f_x.close()
    f_y.close()
    f_info.close()
    if flag_manual:
        f_z.close()


def main():
    args = get_args()
    flag_manual=args.flag_manual
    data_dir = args.data_dir
    out_dir = args.out_dir
    lm_file = args.lm_file
    lm_prob = args.lm_prob
    file_prefix = args.prefix
    start = args.start
    end = args.end
    generate_train_noisy(data_dir, out_dir, file_prefix, lm_file, lm_prob, flag_manual, start, end)


if __name__ == '__main__':
    main()
