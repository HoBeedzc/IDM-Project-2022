# Import module and define global variables
import re
import os
import math
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

PUNCTUATION = "~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"
STEMMER = PorterStemmer()


def is_stop_word(word):
    return word in stopwords.words('english')


def stem_word(word):
    return STEMMER.stem(word)


def filter_line(line):
    return re.sub(r'[{}]+'.format(PUNCTUATION), ' ',
                  line.strip('\n')).strip().lower()


def walk_load_files(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            with open(os.path.join(dirname, filename), errors='ignore') as f:
                yield filename, f


def walk_filter_lines(path):
    for _, f in walk_load_files(path):
        for line in f.readlines():
            yield filter_line(line)


def build_vocabulary_base(base_dir='/kaggle/input/emailtexts/data/emailtexts'):
    vocabulary_base = {}
    for line in walk_filter_lines(base_dir):
        for i in line.split():
            if is_stop_word(i):
                continue
            word = stem_word(i)
            vocabulary_base[word] = vocabulary_base.get(word, 0) + 1
    return vocabulary_base


# compute the tf-idf value
"""
Follow the steps below:
- Calculate the tf value of each word
- Calculate the idf value of each word
- Compute the tf-idf value
"""


def cal_file_tf(vocabulary_base, filehandler):
    res = {}
    all_words = 0
    # init res dict
    for word, _ in vocabulary_base.items():
        res[word] = 0
    for line in filehandler:
        for word in filter_line(line).split():
            all_words += 1
            if is_stop_word(word):
                continue
            word = stem_word(word)
            res[word] += 1
    # compute tf
    for word, count in res.items():
        res[word] = count / all_words
    return res


def tf(vocabulary_base, base_dir):
    tf_dic = {}
    for filename, f in walk_load_files(base_dir):
        tf_dic[filename] = cal_file_tf(vocabulary_base, f)
    # tf_df = pd.DataFrame(tf_dic)
    return tf_dic


def idf(vocabulary_base, tf_dic):
    idf_dic = {}
    for word, _ in vocabulary_base.items():
        count = 0
        file_num = 0
        for _, tf_v in tf_dic.items():
            file_num += 1
            if tf_v[word] == 0:
                continue
            count += 1
        if not count: print(word)
        idf_dic[word] = math.log((1 + file_num) / count)
    return idf_dic


def tf_idf(vocabulary_base,
           base_dir='/kaggle/input/emailtexts/data/emailtexts'):
    tf_dic = tf(vocabulary_base, base_dir)
    idf_dic = idf(vocabulary_base, tf_dic)
    tf_idf_dic = {}
    for filename, tf_vec in tf_dic.items():
        value = {}
        for word, _ in vocabulary_base.items():
            value[word] = tf_vec[word] * idf_dic[word]
        tf_idf_dic[filename] = value
    tfidf_df = pd.DataFrame(tf_idf_dic).T
    return tfidf_df, tf_idf_dic


def main():
    # use tf-idf to preprocess the documents into numerical data.
    BASE_DIR4 = './data/classification'
    # remove mac os index file .DS_Store
    os.system("rm ./data/classification/.DS_Store")
    os.system("rm ./data/classification/baseball/.DS_Store")
    os.system("rm ./data/classification/hockey/.DS_Store")
    tc_vb = build_vocabulary_base(base_dir=BASE_DIR4)
    tc_tfidf_df, tc_tf_idf_dic = tf_idf(vocabulary_base=tc_vb,
                                        base_dir=BASE_DIR4)
    tc_tfidf_df


if __name__ == "__main__":
    main()