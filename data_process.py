import gzip
from os.path import join, exists
from os import listdir, makedirs
import json
from multiprocessing import Pool
import re
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='folder of data.')
    parser.add_argument('--out_dir', type=str, help='folder of output.')
    args = parser.parse_args()
    return args


replace_xml = {'&lt;': '<', '&gt;': '>', '&quot;': '"',
               '&apos;': '\'', '&amp;': '&'}


def process_file(paras):
    fn, out_fn = paras
    with gzip.open(fn, 'rb') as f_:
        content = f_.readlines()
    out_x = open(out_fn + '.x', 'w')                    # output file for OCR'd text
    out_y = open(out_fn + '.y', 'w')                    # output file for duplicated texts (witnesses)
    out_z = open(out_fn + '.z', 'w')                    # output file for manual transcription
    # output file for the information of OCR'd text, each line contains:
    # (group no., line no., file_id, begin index in file, end index in file, number of witnesses, number of manual transcriptions)
    out_x_info = open(out_fn + '.x.info', 'w')
    # output file for the information of each witness, each line contains:
    # (line no, file_id, begin index in file)
    out_y_info = open(out_fn + '.y.info', 'w')
    # output file for the information of each manual transcription,
    # each line contains: (line no, file_id, begin index in file)
    out_z_info = open(out_fn + '.z.info', 'w')
    cur_line_no = 0
    cur_group = 0
    for line in content:
        line = json.loads(line.strip(b'\r\n'))
        cur_id = line['id']
        lines = line['lines']
        for item in lines:
            begin = item['begin']
            text = item['text']                             # get the OCR'd text line
            for ele in replace_xml:
                text = re.sub(ele, replace_xml[ele], text)
            text = text.replace('\n', ' ')                  # remove '\n' and '\t' in the text
            text = text.replace('\t', ' ')
            text = ' '.join([ele for ele in text.split(' ')
                             if len(ele.strip()) > 0])
            if len(text.strip()) == 0:
                continue
            out_x.write(text + '\n')
            wit_info = ''
            wit_str = ''
            man_str = ''
            man_info = ''
            num_manul = 0
            num_wit = 0
            if 'witnesses' in item:
                for wit in item['witnesses']:
                    wit_begin = wit['begin']
                    wit_id = wit['id']
                    wit_text = wit['text']
                    for ele in replace_xml:
                        wit_text = re.sub(ele, replace_xml[ele], wit_text)
                    wit_text = wit_text.replace('\n', ' ')
                    wit_text = wit_text.replace('\t', ' ')
                    wit_text = ' '.join([ele for ele in wit_text.split(' ')
                                         if len(ele.strip()) > 0])
                    if 'manual' in wit_id:                              # get the manual transcription
                        num_manul += 1
                        man_info += str(wit_id) + '\t' + str(wit_begin) + '\t'
                        man_str += wit_text + '\t'
                    else:                                               # get the witnesses
                        num_wit += 1
                        wit_info += str(wit_id) + '\t' + str(wit_begin) + '\t'
                        wit_str += wit_text + '\t'
            if len(man_str.strip()) > 0:
                out_z.write(man_str[:-1] + '\n')
                out_z_info.write(str(cur_line_no) + '\t' + man_info[:-1] + '\n')
            if len(wit_str.strip()) > 0:
                out_y.write(wit_str[:-1] + '\n')
                out_y_info.write(str(cur_line_no) + '\t' + wit_info[:-1] + '\n')
            out_x_info.write(str(cur_group) + '\t' + str(cur_line_no) + '\t' + str(cur_id) + '\t' + str(begin) + '\t' + str(len(text) + begin) + '\t' + str(num_wit) + '\t' + str(num_manul) + '\n')
            cur_line_no += 1
        cur_group += 1
    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


def merge_file(data_dir, out_dir):   # merge all the output files and information files
    list_file = [ele for ele in listdir(data_dir) if ele.startswith('part-')]
    list_out_file = [join(out_dir, 'pair.' + str(i)) for i in range(len(list_file))]
    out_fn = join(out_dir, 'pair')
    out_x = open(out_fn + '.x', 'w')
    out_y = open(out_fn + '.y', 'w')
    out_z = open(out_fn + '.z', 'w')
    out_z_info = open(out_fn + '.z.info', 'w')
    out_x_info = open(out_fn + '.x.info', 'w')
    out_y_info = open(out_fn + '.y.info', 'w')
    last_num_line = 0
    last_num_group = 0
    total_num_y = 0
    total_num_z = 0
    for fn in list_out_file:
        num_line = 0
        with open(fn + '.x') as f_:
            for line in f_:
                out_x.write(line)
                num_line += 1
        with open(fn + '.y') as f_:
            for line in f_:
                out_y.write(line)
        with open(fn + '.z') as f_:
            for line in f_:
                out_z.write(line)
        dict_x2liney = {}
        dict_x2linez = {}
        with open(fn + '.y.info') as f_:
            for line in f_:
                line = line.split('\t')
                line[0] = str(int(line[0]) + last_num_line)
                dict_x2liney[line[0]] = total_num_y
                total_num_y += 1
                out_y_info.write('\t'.join(line))
        with open(fn + '.z.info') as f_:
            for line in f_:
                line = line.split('\t')
                line[0] = str(int(line[0]) + last_num_line)
                dict_x2linez[line[0]] = total_num_z
                total_num_z += 1
                out_z_info.write('\t'.join(line))
        num_group = 0
        with open(fn + '.x.info') as f_:
            for line in f_:
                line = line.strip('\r\n').split('\t')
                cur_group = int(line[0])
                line[0] = str(int(line[0]) + last_num_group)
                line[1] = str(int(line[1]) + last_num_line)
                if line[1] in dict_x2liney:
                    line.append(str(dict_x2liney[line[1]]))
                else:
                    line[5] = '0'
                if line[1] in dict_x2linez:
                    line.append(str(dict_x2linez[line[1]]))
                else:
                    line[6] = '0'
                out_x_info.write('\t'.join(line) + '\n')
                if cur_group > num_group:
                    num_group = cur_group
        last_num_group += num_group
        last_num_line += num_line
        for post_fix in ['.x', '.y', '.z']:
            os.remove(fn + post_fix)
            os.remove(fn + post_fix + '.info')
    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


def process_data(data_dir, out_dir):
    list_file = [ele for ele in listdir(data_dir) if ele.startswith('part')]
    list_out_file = [join(out_dir, 'pair.' + str(i)) for i in range(len(list_file))]
    list_file = [join(data_dir, ele) for ele in list_file]
    if not exists(out_dir):
        makedirs(out_dir)
    # process_file((list_file[0], list_out_file[0]))
    pool = Pool(100)
    pool.map(process_file, zip(list_file, list_out_file))


def main():
    args = get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    process_data(data_dir, out_dir)
    merge_file(data_dir, out_dir)


if __name__ == '__main__':
    main()
