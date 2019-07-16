import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier

from collections import defaultdict
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
import pandas as pd
import numpy as np
import re
from tqdm import tqdm


regex = re.compile(r'[u4e00-u9fa5A-Za-z！？~～。….?!#$￥]')
punctuations = ['.', '…', '！', '？', '!', '?', '。', '#', '$', '￥', '~', '～']
mood_particle = ['嘛', '了', '么', '呢', '吧', '啊', '阿', '啦', '唉', '呢', '吧', '哇', '呀', '吗', '哦', '噢', '喔',
                 '呵', '嘿', '吁', '吓', '吖', '吆', '呜', '咔', '咚', '呼', '呶', '呣', '咝', '咯', '咳', '呗', '咩',
                 '哪', '哎']
sentiment = pd.read_csv('./textdata/BosonNLP_sentiment_score.txt', sep=' ', encoding='utf-8', header=None)
jieba.load_userdict('./textdata/dict.txt')
sen_list = open("./textdata/BosonNLP_sentiment_score.txt", 'r', encoding='utf-8').readlines()
non_list = open("./textdata/neg.txt", 'r', encoding='utf-8').read().splitlines()
degree_list = open("./textdata/degree.txt", 'r', encoding='utf-8').read().splitlines()
sen_dict = defaultdict(float)
for s in sen_list:
    try:
        sen_dict[s.split(' ')[0]] = float(s.split(' ')[1][:-1])
    except:
        pass
non_dict = defaultdict(int)
for d in non_list:
    non_dict[d] = 1
degree_dict = defaultdict(int)
for d in degree_list:
    degree_dict[d.split(',')[0]] = int(d.split(',')[1])

def train_vector(texts):
    model_2 = Word2Vec(size=300, min_count=1)
    model_2.build_vocab(texts)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format("./pretrained/sgns.weibo.word", binary=False, unicode_errors='ignore')
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format("./pretrained/sgns.weibo.word", binary=False, unicode_errors='ignore')
    model_2.train(texts, total_examples=total_examples, epochs=model_2.iter)
    model_2.wv.save_word2vec_format('./pretrained/sgns.emoji.word', binary=False)


def sent2vec(words, model):
    M = []
    for w in words:
        try:
            M.append(model.wv.vectors[model.wv.vocab[w].index])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())



def sentiment_change(words):
    sen = 0.
    non = 0
    degree = 0
    for word in words:
        sen += sen_dict[word]
        non += non_dict[word]
        degree += degree_dict[word]
    return [sen, non, degree]


def train():
    test_f = open("./textdata/test.data", encoding='utf-8')
    test_data = test_f.read().splitlines()
    test_f.close()
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data["num_words"] = test_data["text"].apply(lambda x: len(str(x).split()))
    test_data["num_unique_words"] = test_data["text"].apply(lambda x: len(set(str(x).split())))
    test_data["num_punctuations"] = test_data['text'].apply(lambda x: len([c for c in str(x) if c in punctuations]))
    test_data["mean_word_len"] = test_data["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_data["mood_particle"] = test_data["text"].apply(lambda x: len([c for c in str(x) if c in mood_particle]))
    test_data["split_words"] = test_data["text"].apply(lambda x: [word.lower() for word in jieba.cut(regex.sub(' ', str(x)), cut_all=False) if word.strip()])
    test_data.drop(["text"], axis=1, inplace=True)
    sentiment_score = [sentiment_change(x) for x in tqdm(test_data.split_words)]
    sentiment_score = np.array(sentiment_score)
    test_data = pd.concat([test_data, pd.DataFrame(sentiment_score)], axis=1)

    data = pd.read_csv('train.data.tsv', sep='\t', encoding='utf-8', converters={'label': np.int, 'text': str})
    data.drop(["label"], axis=1, inplace=True)
    data["num_words"] = data["text"].apply(lambda x: len(str(x).split()))
    data["num_unique_words"] = data["text"].apply(lambda x: len(set(str(x).split())))
    data["num_punctuations"] = data['text'].apply(lambda x: len([c for c in str(x) if c in punctuations]))
    data["mean_word_len"] = data["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    data["mood_particle"] = data["text"].apply(lambda x: len([c for c in str(x) if c in mood_particle]))
    data["split_words"] = data["text"].apply(lambda x: [word.lower() for word in jieba.cut(regex.sub(' ', str(x)), cut_all=False) if word.strip()])
    data.drop(["text"], axis=1, inplace=True)
    sentiment_score = [sentiment_change(x) for x in tqdm(data.split_words)]
    sentiment_score = np.array(sentiment_score)
    data = pd.concat([data, pd.DataFrame(sentiment_score)], axis=1)
    texts = data["split_words"].values.tolist() + test_data["split_words"].values.tolist()
    train_vector(texts)
    model = Word2Vec.load('./pretrained/emoji.word')
    test_vectors = [sent2vec(x, model) for x in tqdm(test_data.split_words)]
    test_vectors = np.array(test_vectors)
    test_data = pd.concat([test_data, pd.DataFrame(test_vectors)], axis=1)
    vectors = [sent2vec(x, model) for x in tqdm(data.split_words)]
    vectors = np.array(vectors)
    data = pd.concat([data, pd.DataFrame(vectors)], axis=1)
    del test_vectors
    del vectors
    tfidf = TfidfVectorizer()
    train_texts = data["split_words"].apply(lambda x: ' '.join(x)).values.tolist()
    test_texts = test_data["split_words"].apply(lambda x: ' '.join(x)).values.tolist()
    full_tfidf = tfidf.fit_transform(train_texts + test_texts)
    train_tfidf = tfidf.transform(train_texts)
    svd_obj = TruncatedSVD(n_components=20, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    train_svd.columns = ['svd_char_' + str(i) for i in range(20)]
    data = pd.concat([data, train_svd], axis=1)
    del train_svd
    data.drop(["split_words"], axis=1, inplace=True)
    data.to_csv('train_data.tsv', sep='\t', encoding='utf-8', index=None)
    test_tfidf = tfidf.transform(test_texts)
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    test_svd.columns = ['svd_char_' + str(i) for i in range(20)]
    test_data = pd.concat([test_data, test_svd], axis=1)
    del test_svd
    test_data.drop(["split_words"], axis=1, inplace=True)
    test_data.to_csv('test_data.tsv', sep='\t', encoding='utf-8', index=None)

if __name__ == '__main__':
    train()