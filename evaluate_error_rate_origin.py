from levenshtein import align_pair
from multiprocessing import Pool
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path of the file to evaluate.')
    parser.add_argument('--gt', type=str, help='Path of ground truth data.')
    parser.add_argument('--lowercase', type=bool, default=True, help='Whether to lowercase the input and ground truth before evaluation.')
    parser.add_argument('--char', type=bool, default=True, help='Whether to evaluate char or word error rate.')
    args = parser.parse_args()
    return args


def error_rate(dis_xy, len_y):
    dis_xy = np.asarray(dis_xy)
    len_y = np.asarray(len_y)
    micro_error = np.mean(dis_xy/len_y)
    macro_error = np.sum(dis_xy) / np.sum(len_y)
    return micro_error, macro_error


def evaluate(args,):
    with open(args.input, encoding='utf-8') as f_:
        list_x = [ele.strip('\n').split('\t')[0] for ele in f_.readlines()]
        if args.lowercase:
            list_x = [ele.lower() for ele in list_x]
    with open(args.gt, encoding='utf-8') as f_:
        list_y = [ele.strip('\n').split('\t')[0] for ele in f_.readlines()]
        if args.lowercase:
            list_y = [ele.lower() for ele in list_y]
    if args.char:
        len_y = [len(y) for y in list_y]
    else:
        len_y = [len(y.split()) for y in list_y]
    print(len(len_y))
    pool = Pool(100)
    dis_xy = align_pair(pool, list_y, list_x, flag_char=args.char)
    micro_error, macro_error = error_rate(dis_xy, len_y)
    if args.char:
        print('Micro average of char error rate: %.6f' % micro_error)
        print('Macro average of char error rate: %.6f' % macro_error)
    else:
        print('Micro average of word error rate: %.6f' % micro_error)
        print('Macro average of word error rate: %.6f' % macro_error)


def main():
    args = get_args()
    evaluate(args)


if __name__ == '__main__':
    main()
