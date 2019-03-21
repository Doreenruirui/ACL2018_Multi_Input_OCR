import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='folder of data.')
    parser.add_argument('--out_dir', type=str, help='folder of output.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training and test data.')
    args = parser.parse_args()
    return args


def generate_train_supervised(paras):
    if not os.path.exists(paras.out_dir):
        os.makedirs(paras.out_dir)
    with open(os.path.join(paras.data_dir, 'pair.z'), encoding='utf-8') as f_:
        lines = f_.readlines()
    with open(os.path.join(paras.data_dir, 'pair.z.info'), encoding='utf-8') as f_:
        lines_info = f_.readlines()
    with open(os.path.join(paras.data_dir, 'pair.x'), encoding='utf-8') as f_:
        lines_x = f_.readlines()
    nline = len(lines)
    index = np.arange(nline)
    np.random.shuffle(index)
    dict_index = {}
    ntest = np.int(np.round(nline * 0.2))
    dict_index['test'] = index[-ntest:]
    ndev = np.int(np.round((nline - ntest) * 0.2))
    dict_index['dev'] = index[-ndev-ntest:-ntest]
    dict_index['train'] = index[:-ntest-ndev]
    for dataset in ['train', 'test', 'dev']:
        with open(os.path.join(paras.out_dir, dataset + '.y.txt'), 'w', encoding='utf-8') as f_:
            for lid in dict_index[dataset]:
                f_.write(lines[lid].strip('\n').split('\t')[0] + '\n')
        with open(os.path.join(paras.out_dir, dataset + '.info.txt'), 'w', encoding='utf-8') as f_:
            for lid in dict_index[dataset]:
                f_.write(lines_info[lid])
        with open(os.path.join(paras.out_dir, dataset + '.x.txt'), 'w', encoding='utf-8') as f_:
            for lid in dict_index[dataset]:
                cur_x_lid = int(lines_info[lid].strip().split('\t')[0])
                f_.write(lines_x[cur_x_lid])


def main():
    args = get_args()
    generate_train_supervised(args)


if __name__ == '__main__':
    main()





