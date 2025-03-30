import pandas as pd
import os
import jieba
import string
import re

class textProc():
    def __init__(self):
        pass

    def divide_into_chapter(self, filename):
        df = pd.read_excel(filename, engine='openpyxl', header=None)
        chaps = df.iloc[:, 2].tolist()
        for idx, tx in enumerate(chaps):
            fo = open(f'mychap/{filename[5:-5]}_{idx+1}.txt', 'w', encoding='utf-8')
            fo.write(tx)
            fo.close()

    def perform_segmentation(self):
        files = os.listdir('mytext')
        for file in files:
            with open(f'mychap/{file}', 'r', encoding='utf-8') as fi:
                with open(f'mychapdivs/{file}', 'w', encoding='utf-8') as fo:
                    line = fi.readline()
                    seg_list = jieba.cut(line, cut_all=False)
                    words = " ".join(seg_list)
                    fo.write(words)

    def perform_wordcount(self):
        files = os.listdir('mychapdivs')
        for file in files:
            with open(f'mychapdivs/{file}', 'r', encoding='utf-8') as fi:
                res_dict = {}
                line = fi.readline()
                delset = str.maketrans('', '', string.punctuation)
                line = line.translate(delset)
                line = ''.join(line.split('\n'))
                line = self.sub_replace(line)
                words = line.split()
                for word in words:
                    res_dict[word] = res_dict.get(word, 0) + 1
                with open(f'mywc/{file}', 'w', encoding='utf-8') as fo:
                    sorted_res = sorted(res_dict.items(), key=lambda d:d[1], reverse=True)
                    for one in sorted_res:
                        line = ''.join(f'{one[0]}\t{one[1]}\n')
                        fo.write(line)

    def sub_replace(self, line):
        regex = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]")
        return regex.sub('', line)



if __name__ == "__main__":
    tP = textProc()
    filename = 'text/80.xlsx'
    tP.divide_into_chapter(filename=filename)
    filename = 'text/40.xlsx'
    tP.divide_into_chapter(filename=filename)
    filename = 'text/Guiyou.xlsx'
    tP.divide_into_chapter(filename=filename)
    tP.perform_segmentation()
    tP.perform_wordcount()