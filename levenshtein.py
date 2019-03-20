import numpy as np
# from multiprocessing import Pool

output = None
output_str = None


def align(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    d = np.ones((len1 + 1, len2 + 1), dtype=int) * 1000000
    op = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        d[i, 0] = i
        op[i, 0] = 2
    for j in range(len2 + 1):
        d[0, j] = j
        op[0, j] = 1
    op[0, 0] = 0
    for i in range(1, len1 + 1):
        char1 = str1[i - 1]
        for j in range(1, len2 + 1):
            char2 = str2[j - 1]
            if char1 == char2:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i, j - 1] + 1, d[i - 1, j] + 1, d[i - 1, j - 1] + 1)
                if d[i, j] == d[i, j - 1] + 1:
                    op[i, j] = 1
                elif d[i, j] == d[i - 1, j] + 1:
                    op[i, j] = 2
                elif d[i, j] == d[i - 1, j - 1] + 1:
                    op[i, j] = 3
    return d[len1, len2]


def align_one2many_thread(para):
    thread_num, str1, list_str, flag_char, flag_low = para
    str1 = ' '.join([ele for ele in str1.split(' ') if len(ele) > 0])
    if flag_low:
        str1 = str1.lower()
    min_dis = float('inf')
    min_str = ''
    for i in range(len(list_str)):
        cur_str = ' '.join([ele for ele in list_str[i].split(' ') if len(ele) > 0])
        if flag_low:
            cur_str = cur_str.lower()
        if not flag_char:
            dis = align(str1.split(), cur_str.split())
        else:
            dis = align(str1, cur_str)
        if dis < min_dis:
            min_dis = dis
            min_str = list_str[i]
    return min_dis


def align_one2one(para):
    thread_num, str1, str2, flag_char, flag_low = para
    if flag_low:
        str1 = str1.lower()
        str2 = str2.lower()
    if flag_char:
        return align(str1, str2)
    else:
        return align(str1.split(), str2.split())


def align_pair(P, truth, cands, flag_char=1, flag_low=1):
    global output, output_str
    ndata = len(truth)
    output = [0 for _ in range(ndata)]
    list_index = np.arange(ndata).tolist()
    list_flag = [flag_char for _ in range(ndata)]
    list_low = [flag_low for _ in range(ndata)]
    paras = zip(list_index, truth, cands, list_flag, list_low)
    results = P.map(align_one2one, paras)
    return results


def align_beam(P, truth, cands, flag_char=1, flag_low=1):
    global output, output_str
    ndata = len(truth)
    list_index = np.arange(ndata).tolist()
    list_flag_char = [flag_char for _ in range(ndata)]
    list_flag_low = [flag_low for _ in range(ndata)]
    paras = zip(list_index, truth, cands, list_flag_char, list_flag_low)
    results = P.map(align_one2many_thread, paras)
    return results
