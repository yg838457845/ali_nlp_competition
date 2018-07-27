
# coding: utf-8

# In[1]:



import numpy as np
import sys, io, re, time, jieba.analyse
from gensim.models import word2vec
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 

STOP_WORDS = []  # 停用词
fstop = io.open('chinese_stopwords.txt', 'r', encoding='utf-8')
for eachWord in fstop:
    STOP_WORDS.append(eachWord.strip())

simiwords = {}
with io.open("simiwords.txt", encoding='utf-8') as fr:
    for line in fr:
        words = re.split(",", line.strip())
        simiwords[words[0]] = words[1]

def process_simi_stop(simiwords, stopwords, line):
    line = str(line).lower()
    for word, subword in simiwords.iteritems():
        if word in line:
            # print line
            #line = re.sub(word, subword, line)
            line = line.replace(word,subword)
            # print subword
    line = line.replace("。", "").replace('嚒', '么').replace('蜜码', '密码').replace('注消', '注销').replace('不了', '不能')
    words1 = [w for w in jieba.cut(line) if w.strip()]
    word1 = []
    for i in words1:
        if i not in stopwords:
            word1.append(i)
    return word1,line
        

def splitSentence(inputFile, inpath, segment, submit):
    print u'分词开始！', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # 输出当前时间
    start = time.clock()
    jieba.load_userdict("jieba_dict.txt")
    corpus = []

    simiwords = {}
    with io.open("simiwords.txt", encoding='utf-8') as fr:
        for line in fr:
            words = re.split(",", line.strip())
            simiwords[words[0]] = words[1]

    stopwords = []  # 停用词
    fstop = open('chinese_stopwords.txt', 'r')
    for eachWord in fstop:
        stopwords.append(eachWord.strip())
        # print eachWord.strip()

    fin = open(inputFile, 'r')  # 以读的方式打开文件 inputFile
    fin1 = open('sentences_1.txt','w')
    for eachLine in fin:
        eachLine = re.sub("\*", " ", eachLine)
        lineno, sen1, sen2, label = eachLine.strip().split('\t')
        word1,sen_1 = process_simi_stop(simiwords, stopwords, sen1)
        word2,sen_2 = process_simi_stop(simiwords, stopwords, sen2)
        fin1.write(sen_1)
        fin1.write('\n')
        fin1.write(sen_2)
        fin1.write('\n')
        corpus.append(word1)
        corpus.append(word2)
    print "行数的二倍", len(corpus)
    fin.close()
    fin1.close()
    with open(segment, 'w') as fs:
        for word in corpus:
            # print type(word)
            for w in word:
                # print w
                fs.write(w)  # 将分词好的结果写入到输出文件
                fs.writelines(' ')
            fs.write('\n')
    end = time.clock()
    print u'分词实际用时：', end - start
    return corpus

def word22vec(inpath, testpath, filename, vec_dim):
    # jieba 分词
    splitSentence(testpath,inpath, filename, False)

    # 训练词向量模型
    sentences = word2vec.Text8Corpus(filename)
    model = word2vec.Word2Vec(sentences, sg=1, size=vec_dim, window=5, min_count=10, negative=3, sample=0.001, hs=1,
                              workers=4)
    model.save('word2vec_model')  # save
    print(u'词向量训练完毕', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  # 输出当前时间
    return model


# In[2]:



from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import sys
import time
import jieba
import io
import re
# import distance
# from fuzzywuzzy import fuzz
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')


def combine(combine_file, filename1, filename2):
    len_merge_sum = 0
    with open(combine_file, 'w') as fout:
        with open(filename1, 'r') as f1:
            for eachLine in f1:
                lineno, sen1, sen2, label = eachLine.strip().split('\t')
                fout.write(lineno + '\t' + sen1 + '\t' + sen2 + '\t' + label + '\n')
                len_merge_sum += 1
        with open(filename2, 'r') as f1:
            for eachLine in f1:
                lineno, sen1, sen2, label = eachLine.strip().split('\t')
                fout.write(lineno + '\t' + sen1 + '\t' + sen2 + '\t' + label + '\n')
                len_merge_sum += 1
    fout.close()
    return combine_file, len_merge_sum



def filter_word_in_model(model, filename):
    a = []
    with open(filename, 'r') as file_to_read:
        for line in file_to_read:
            if True:
                if not line:
                    break
                a.append(line)
    sentences = []  # 读sentences 里面的词
    for i in range(len(a)):
        b = a[i].strip().split()
        sentences.append(b)
    print 'sentences length:', len(sentences)
    new_sentences = []  # 完成获取模型训练，剩余含有词向量序列
    for i in range(len(sentences)):
        new_sentence = []
        for j in range(len(sentences[i])):
            if sentences[i][j].decode('utf8') in model:
                new_sentence.append(sentences[i][j])
        new_sentences.append(new_sentence)
    print 'new_sentences length: ', len(new_sentences)
    # print new_sentences[:3]
    # print(np.array(new_sentences).shape)
    print u'new_sentences,用时', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # 输出当前时间
    with open('new_sentences.txt', 'w') as fs:  # 写入new_sentences
        for word in new_sentences:
            for w in word:
                fs.write(w)  # 将分词好的结果写入到输出文件
                fs.writelines(' ')
            fs.write('\n')
    return new_sentences


SAFE_DIV = 0.0001


def preprocess(x):
        x = str(x).lower()
        for word, subword in simiwords.iteritems():
            if word in x:
                x = x.replace(word, subword)
        x = x.replace("。", "").replace('嚒', '么').replace('蜜码', '密码').replace('注消', '注销').replace('不了', '不能')
        jieba.load_userdict("jieba_dict.txt")
        words1 = [w.decode('utf8') for w in jieba.cut(x) if w.strip()]
        word1 = []
        for i in words1:
            if i not in STOP_WORDS:
                word1.append(i)
        return word1


def get_token_features(q1, q2):
    token_features = [0.0]*10
    q1_tokens = q1
    # print(q1_tokens)
    q2_tokens = q2

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features


def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(str(a), str(b)))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)


def extract_features(df):
    df["q1"] = df["q1"].fillna("").apply(preprocess)
    df["q2"] = df["q2"].fillna("").apply(preprocess)
    # print(type(df["q1"]))
    # print(df["q1"])

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["q1"], x["q2"]), axis=1)
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
    return df


print('Extracting features for train:')

filename1 = 'atec_nlp_sim_train.csv'
filename2 = 'atec_nlp_sim_train_add.csv'
combine_file, len_merge_sum = combine('merge_sum.csv', filename1, filename2)

SUBMIT = True
if SUBMIT:
    inpath, outpath = sys.argv[1], sys.argv[2]
    trainpath = combine_file
else:
    inpath, outpath = 'merge_10%_without_label.csv', 'output.csv'
    trainpath = 'merge_90%.csv'
    test_num = 92228

train_df = pd.read_table(trainpath, encoding='utf8', names=(['id', 'q1', 'q2', 'label']))
print(train_df[:3])
train_df = extract_features(train_df)
train_df.drop(['id', 'label'], axis=1, inplace=True)
train_df.to_csv("nlp_features_train.csv", encoding='utf8', index=False)
print(train_df.shape)

test_df = pd.read_table(inpath, encoding='utf8', names=(['id', 'q1', 'q2']))
print(test_df[:3])
test_df = extract_features(test_df)
test_df.drop(['id'], axis=1, inplace=True)
test_df.to_csv("nlp_features_test.csv", encoding='utf8', index=False)
print(test_df.shape)



def get_tiidf_vec(filename):
    corpus = [' '.join(a) for a in filename]
    vectorizer = CountVectorizer(min_df=0, token_pattern=r"(?u)\W{1}|(?u)\b\w+\b")
    result = vectorizer.fit_transform(corpus)  # 文本转为词频矩阵
    transformer = TfidfTransformer()  # 统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(result)  # fit_transform是计算tf-idf
    vecs = []  # 每一个值的tfidf值
    weight = tfidf.toarray()
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    print len(word)
    print weight.shape

    vecs = []  # 每一个值的tfidf值
    for val in range(len(filename)):
        b = tfidf[val].toarray()[0]
        vec1 = []
        if len(b) == len(filename[val]):
            for i in range(len(b)):
                vec1.append(b[i])
            vecs.append(vec1)
        else:
            f = filename[val]
            for i in range(len(f)):
                r = word.index(f[i])
                vec1.append(weight[val][r])
            vecs.append(vec1)
    for i in range(len(vecs)):
        if len(vecs[i]) == len(filename[i]):
            pass
        else:
            print(i)
    return vecs


def transform_vec(new_sentences, model, size, tfidf_Vec):
    vec_titles = []  # 获取句子的向量
    for val in range(len(new_sentences)):
        vec = np.zeros(shape=(1, size))
        for i in range(len(new_sentences[val])):
            vec += np.array(model[new_sentences[val][i].decode('utf8')])*tfidf_Vec[val][i]
        vec_titles.append(vec)
    print(np.array(vec_titles).shape)
    vec_titles = list(map(lambda x: x[0], vec_titles))  # 去掉外部的[], 获得title 的向量形式
    np.save("train_data_title_vec.npy", vec_titles)
    print u'生成train_data_title_vec完毕', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # 输出时间
    return vec_titles


def eval_file(label1, pre):
    tp, tn, fp, fn = 0, 0, 0, 0
    for la, pr in zip(label1, pre):
        if la == 1 and pr == 1:
            tp += 1
        elif la == 1 and pr == 0:
            fn += 1
        elif la == 0 and pr == 0:
            tn += 1
        elif la == 0 and pr == 1:
            fp += 1
    recall = float(tp)/float(tp+fn)
    precision = float(tp)/float(tp+fp)
    f11 = 2*recall*precision/(recall+precision)
    return f11


def cos_Vector(x, y):  # 用cos求夹角
    if len(x) != len(y):
        print u'error input,x and y is not in the same space'
        return
    x = np.array(x)
    y = np.array(y)
    num = (x * y.T)
    num = float(num.sum())
    if num == 0:
        return 0
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def vec_minus(x, y):  # 相加
    if len(x) != len(y):
        print u'error input,x and y is not in the same space'
        return
    x = np.array(x)
    y = np.array(y)
    sim = x - y
    return sim


def vec_multi(x, y):  # 相乘
    if len(x) != len(y):
        print u'error input,x and y is not in the same space'
        return
    x = np.array(x)
    y = np.array(y)
    sim1 = x * y
    return sim1


def calEuclideanDistance(x, y):
    if len(x) != len(y):
        print u'error input,x and y is not in the same space'
        return
    dist = np.sqrt(np.sum(np.square(x - y)))
    return dist


def cal_jaccard(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    avg_len = (len(set1) + len(set2)) / 2
    min_len = min(len(set1), len(set2))
    # return len(set1 & set2) * 1.0 / (len(set1) + len(set2) - len(set1 & set2))
    return len(set1 & set2) * 1.0 / min_len


# In[3]:



import re
import pandas as pd
import numpy as np
import io
import jieba
# import functions
import tensorflow as tf
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
from keras import optimizers
# from nltk.stem.wordnet import WordNetLemmatizer
import sys
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 


np.random.seed(0)

MAX_SEQUENCE_LENGTH = 30
MIN_WORD_OCCURRENCE = 20
REPLACE_WORD = "图腾"
EMBEDDING_DIM = 150
NUM_FOLDS = 10
BATCH_SIZE = 256
# EMBEDDING_FILE = "glove.840B.300d.txt"   # 词向量

STOP_WORDS = []  # 停用词
fstop = open('chinese_stopwords.txt', 'r')
for eachWord in fstop:
    STOP_WORDS.append(eachWord.strip())
simiwords = {}
with io.open("simiwords.txt", encoding='utf-8') as fr:
    for line in fr:
        words = re.split(",", line.strip())
        simiwords[words[0]] = words[1]


def is_numeric(s):
    return any(i.isdigit() for i in s)

#以句子为输入，分割后的数组为输出
def preprocess(x):
    x = str(x).lower()
    for word, subword in simiwords.iteritems():
        if word in x:
            x = x.replace(word, subword)
    x = x.replace("。", "").replace('嚒', '么').replace('蜜码', '密码').replace('注消', '注销').replace('不了', '不能')
    jieba.load_userdict("jieba_dict.txt")
    words1 = [w.decode('utf8') for w in jieba.cut(x) if w.strip()]
    return words1

#以句子列表为输入，返回new_q——高频词列表，surplus_q低频词列表，numbers_q数字列表
def prepare(q):
    new_q = []
    new_qs = []
    surplus_q = []
    numbers_q = []
    new_memento = True
    # for w in q.split()[::-1]:
    for w in q:
        #top_frequence
        if w in top_words:
            # print w
            new_q = [w] + new_q
            new_memento = True
        elif w not in STOP_WORDS:
            if new_memento:
                new_q = ["图腾"] + new_q
                new_memento = False
            if is_numeric(w):
                numbers_q = [w] + numbers_q
            else:
                surplus_q = [w] + surplus_q#save the low_frequence_words
        else:
            # print w
            new_memento = True
        if len(new_q) == MAX_SEQUENCE_LENGTH:
            break
        # print new_q
    new_qs.append(new_q)
    return new_q, set(surplus_q), set(numbers_q)

#输入df为pandas列表
def extract_features(df):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df), 4))
    #q1,q2 表示遍历每一行两个句子对应的数组
    for i, (q1, q2) in enumerate(list(zip(df["q1"], df["q2"]))):
        # print q1, q2
        q1s[i], surplus1, numbers1 = prepare(q1)
        q2s[i], surplus2, numbers2 = prepare(q2)
        features[i, 0] = len(surplus1.intersection(surplus2))
        features[i, 1] = len(surplus1.union(surplus2))
        features[i, 2] = len(numbers1.intersection(numbers2))
        features[i, 3] = len(numbers1.union(numbers2))
    return q1s, q2s, features


SUBMIT = True
if SUBMIT:
    inpath, outpath = sys.argv[1], sys.argv[2]
    trainpath = combine_file
    test_num = len_merge_sum
else:
    inpath, outpath = 'merge_10%_without_label.csv', 'output.csv'
    trainpath = 'merge_90%.csv'
    test_num = 92228
size = 150

filename = 'sentences.txt'

# # 训练词向量模型
model = word22vec(inpath, trainpath, filename, size)
new_sentences = filter_word_in_model(model, filename)
new_sentences = open('new_sentences.txt', 'r').readlines()

texts = []  # 训练集所有句子
for i in new_sentences:
    texts.append(i)
print('Found %s texts.' % len(texts))

# 储存所有的单词
word_set = set()
for i in new_sentences:
    for w in i.strip().split(' '):
        word_set.add(w)
print 'word_set length', len(word_set)

print("Creating the vocabulary of words occurred more than", MIN_WORD_OCCURRENCE)
# 去掉频率小于20的词
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
vectorizer.fit(texts)
top_words = set(vectorizer.vocabulary_.keys())
top_words.add(REPLACE_WORD)
print 'top_words length', len(top_words)  # 1905 # set

# 输出一个字典：一个高频词，对应一个300维词向量
embeddings_index = {}
for word in top_words:
    if word.decode('utf8') in model:
        embeddings_index[word.decode('utf8')] = model[word.decode('utf8')]
    else:
        embeddings_index[word.decode('utf8')] = np.zeros((1, size))
print(len(embeddings_index.keys()))  # 1905*size # list
print("Words are not found in the embedding:", top_words - set(embeddings_index.keys()))
top_words = embeddings_index.keys()


train = pd.read_table(trainpath, encoding='utf8', names=(['id', 'q1', 'q2', 'label']))
test = pd.read_table(inpath, encoding='utf8', names=(['id', 'q1', 'q2']))

train["q1"] = train["q1"].fillna("").apply(preprocess)
train["q2"] = train["q2"].fillna("").apply(preprocess)
print("提取训练集特征，高频词以及低频词特征")
q1s_train, q2s_train, train_q_features = extract_features(train)  # train_q_features  = 4
print len(q1s_train), len(q2s_train)

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
word_index = tokenizer.word_index#高频词
# 第一列高频词的序号
data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=MAX_SEQUENCE_LENGTH)
# 第二列高频词的序号
data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(train["label"])

# embedding_matrix为高频词向量组成的矩阵
nb_words = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.decode('utf8'))
    # print np.array(embedding_vector).shape
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)  # 2756*256
np.save("embedding_matrix.npy", embedding_matrix)
embedding_matrix = np.load("embedding_matrix.npy")

print("Train features are being merged with NLP and Non-NLP features...")
train_nlp_features = pd.read_csv("nlp_features_train.csv")
train_nlp_features.drop(['q1', 'q2', 'csc_min', 'csc_max'], axis=1, inplace=True)
print train_nlp_features.shape, train_q_features.shape
features_train = np.hstack((train_q_features, train_nlp_features))
print 'features_train.shape', features_train.shape

print("Same steps are being applied for test...")
test["q1"] = test["q1"].fillna("").apply(preprocess)
test["q2"] = test["q2"].fillna("").apply(preprocess)
q1s_test, q2s_test, test_q_features = extract_features(test)
test_data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), maxlen=MAX_SEQUENCE_LENGTH)
test_nlp_features = pd.read_csv("nlp_features_test.csv")
test_nlp_features.drop(['q1', 'q2', 'csc_min', 'csc_max'], axis=1, inplace=True)
features_test = np.hstack((test_q_features, test_nlp_features))
print 'feature_test.shape', features_test.shape


#
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + true_positives)
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + true_positives)
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper
#


skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
model_count = 0

# for idx_train, idx_val in skf.split(train["label"], train["label"]):
#     print("MODEL:", model_count)
#     data_1_train = data_1[idx_train]  # print 'data_1_train', data_1_train
#     data_2_train = data_2[idx_train]
#     labels_train = labels[idx_train]
#     f_train = features_train[idx_train]

#     data_1_val = data_1[idx_val]
#     data_2_val = data_2[idx_val]
#     labels_val = labels[idx_val]
#     f_val = features_train[idx_val]

#     embedding_layer = Embedding(nb_words,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SEQUENCE_LENGTH,
#                                 trainable=False)
#     lstm_layer = LSTM(75, recurrent_dropout=0.2)

#     sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
#     embedded_sequences_1 = embedding_layer(sequence_1_input)
#     x1 = lstm_layer(embedded_sequences_1)

#     sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
#     embedded_sequences_2 = embedding_layer(sequence_2_input)
#     y1 = lstm_layer(embedded_sequences_2)

#     features_input = Input(shape=(f_train.shape[1],), dtype="float32")
#     features_dense = BatchNormalization()(features_input)
#     features_dense = Dense(200, activation="relu")(features_dense)
#     features_dense = Dropout(0.2)(features_dense)

#     addition = add([x1, y1])
#     minus_y1 = Lambda(lambda x: -x)(y1)
#     merged = add([x1, minus_y1])
#     merged = multiply([merged, merged])
#     merged = concatenate([merged, addition])
#     merged = Dropout(0.4)(merged)

#     merged = concatenate([merged, features_dense])
#     merged = BatchNormalization()(merged)
#     merged = GaussianNoise(0.1)(merged)

#     merged = Dense(150, activation="relu")(merged)
#     merged = Dropout(0.2)(merged)
#     merged = BatchNormalization()(merged)

#     out = Dense(1, activation="sigmoid")(merged)

#     model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)

#     precision = as_keras_metric(tf.metrics.precision)
#     recall = as_keras_metric(tf.metrics.recall)
#     Nadam = optimizers.Nadam(lr=0.01)
#     model.compile(loss="binary_crossentropy",
#                   optimizer=Nadam, metrics=['accuracy', precision, recall])
#     early_stopping = EarlyStopping(monitor="val_loss", patience=3)
#     best_model_path = "best_model.h5"
#     model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

#     try:
#         model.load_weights(best_model_path)
#     except:
#         pass
#     hist = model.fit([data_1_train, data_2_train, f_train], labels_train,
#                      validation_data=([data_1_val, data_2_val, f_val], labels_val),
#                      epochs=10, batch_size=BATCH_SIZE, shuffle=True,
#                      callbacks=[early_stopping, model_checkpoint], verbose=2)
    
#     print(model_count, "validation loss:", min(hist.history["val_loss"]))

#     preds = model.predict([test_data_1, test_data_2, features_test], batch_size=BATCH_SIZE, verbose=1)
#     submission = pd.DataFrame({"test_id": test["id"], "label": preds.ravel()})

#     preds_offline = preds
#     SUBMIT = False
#     if not SUBMIT:
#         with open('merge_10%.csv', 'r') as f:
#             y_true_10 = []
#             for eachLine in f:
#                 lineno, sen1, sen2, label = eachLine.strip().split('\t')
#                 a = int(label)
#                 y_true_10.append(a)
            
#         N = 200
#         score_best = 0
#         preds = []
#         pred_0_5 = []
#         n_max = 0
#         for thred in range(1, N):  # 阈值的选取，如何找到最好的阈值
#             thred = thred * (np.max(preds_offline) - np.min(preds_offline)) / (1.1*N) + np.min(preds_offline)
#             pred = []
#             for i in range(len(preds_offline)):
#                 if preds_offline[i] > thred:
#                     pred.append(1)
#                 else:
#                     pred.append(0)
#             score = eval_file(y_true_10, pred)
#             if score > score_best:
#                 score_best = score
#                 thred_best = thred
#         print u'最优阈值：', thred_best
#         thred_best_list = []
#         thred_best_list.append(thred_best)
#         np.save('thred_best_list.npy', thred_best_list)

#         for i in range(len(preds_offline)):
#             if preds_offline[i] > thred_best:
#                 preds.append(1)
#             else:
#                 preds.append(0)
#         optimal_f1 = []
#         a = eval_file(y_true_10, preds)
#         optimal_f1.append(a)
#         print 'F1 score is :', a
#         np.save('optimal_f1.npy', optimal_f1)

#     submission.to_csv("prediction/preds_" + str(model_count) + ".csv", index=False)

#     model_count += 1
#     if model_count == 2:
#         break

# In[5]:



embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
lstm_layer = LSTM(75, recurrent_dropout=0.2)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

features_input = Input(shape=(features_test.shape[1],), dtype="float32")
features_dense = BatchNormalization()(features_input)
features_dense = Dense(200, activation="relu")(features_dense)
features_dense = Dropout(0.2)(features_dense)

addition = add([x1, y1])
minus_y1 = Lambda(lambda x: -x)(y1)
merged = add([x1, minus_y1])
merged = multiply([merged, merged])
merged = concatenate([merged, addition])
merged = Dropout(0.4)(merged)

merged = concatenate([merged, features_dense])
merged = BatchNormalization()(merged)
merged = GaussianNoise(0.1)(merged)

merged = Dense(150, activation="relu")(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

out = Dense(1, activation="sigmoid")(merged)

for i in range(NUM_FOLDS):
    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)
    model.load_weights("best_model" + str(i) + ".h5")
    # model.load_weights(best_model_path)
    preds = model.predict([test_data_1, test_data_2, features_test], batch_size=BATCH_SIZE, verbose=1)
    submission = pd.DataFrame({"test_id": test["id"], "label": preds.ravel()})
    preds_offline = preds
    submission.to_csv("predictions/preds_" + str(i) + ".csv", index=False)

program3 = True
if program3:
    # -*- coding: utf-8 -*-
    import numpy as np
    import sys
    import pandas as pd
    import time
    reload(sys)
    sys.setdefaultencoding('utf-8')

    NUM_FOLDS = 10
    predicts = pd.DataFrame()
    for i in range(NUM_FOLDS):
        a = pd.read_csv("predictions/preds_" + str(i) + ".csv")
        predicts[str(i)] = a['label']
    predicts['Col_sum'] = predicts.apply(lambda x: x.sum()/NUM_FOLDS, axis=1)
    preds_offline = predicts['Col_sum']
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        line_id = []
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            line_id.append(lineno)
        for i in range(len(line_id)):
            if preds_offline[i] >= 0.3:
                fout.write(line_id[i] + '\t1\n')
            else:
                fout.write(line_id[i] + '\t0\n')


