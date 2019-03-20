from levenshtein import align_pair, align_beam
from multiprocessing import Pool
import numpy as np

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path of the file to evaluate.')
    parser.add_argument('--gt', type=str, help='Path of ground truth data.')
    parser.add_argument('--beam_size', type=int, default=128, help='The beam size.')
    parser.add_argument('--lowercase', type=bool, default=True, help='Whether to lowercase the input and ground truth before evaluation.')
    parser.add_argument('--char', type=bool, default=True, help='Whether to evaluate char or word error rate.')
    args = parser.parse_args()
    return args


def error_rate(dis_xy, len_y):
    macro_error = np.mean(dis_xy/len_y)
    micro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate(args, beam_size=100):
    line_id = 0
    list_dec = []
    list_beam = []
    list_top = []
    with open(args.input) as f_:
        for line in f_:
            line_id += 1
            cur_str = line.strip()
            if args.lowercase:
                cur_str = cur_str.lower()
            if line_id % beam_size == 1:
                if len(list_beam) == beam_size:
                    list_dec.append(list_beam)
                    list_beam = []
                list_top.append(cur_str)
            list_beam.append(cur_str)
    list_dec.append(list_beam)

    with open(args.gt, 'r') as f_:
        list_y = [ele.strip('\n').split('\t')[0].strip() for ele in f_.readlines()]
        if args.lowercase:
            list_y = [ele.lower() for ele in list_y]
    if args.char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print(len(len_y))
    nthread = 100
    pool = Pool(nthread)
    dis_by = align_beam(pool, list_y, list_dec, flag_char=args.char, flag_low=args.lowercase)
    dis_ty = align_pair(pool, list_y,  list_top, flag_char=args.char, flag_low=args.lowercase)
    micro_error, macro_error = error_rate(dis_ty, len_y)
    best_micro_error, best_macro_error = error_rate(dis_by, len_y)
    if args.char:
        print('Micro average of char error rate: %.6f ' % micro_error)
        print('Macro average of char error rate: %.6f' % macro_error)
        print('Oracle micro average of char error rate: %.6f' % best_micro_error)
        print('Oracle macro average of char error rate: %.6f'% best_macro_error)
    else:
        print('Micro average of word error rate: %.6f' % micro_error)
        print('Macro average of word error rate: %.6f' % macro_error)
        print('Oracle micro average of word error rate: %.6f' % best_micro_error)
        print('Oracle macro average of word error rate: %.6f' % best_macro_error)


def main():
    args = get_args()
    evaluate(args)


if __name__ == '__main__':
    main()
