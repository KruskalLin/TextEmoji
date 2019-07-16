import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import re


models = [('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=1.0), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=1.0), method='isotonic')),
          ('Logit', LogisticRegression(C=30, verbose=1))]

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z！？~～。….]')

def delete_cat(dataset, label, cats):
    new_dataset = []
    new_label = []
    for i in tqdm(range(len(label)), desc='delete'):
        if label[i] not in cats:
            new_dataset.append(dataset[i])
            new_label.append(label[i])
    return new_dataset, new_label

def merge_cat(dataset, label, cats, newcat):
    new_dataset = dataset
    new_label = label
    for i in tqdm(range(len(label)), desc='merge'):
        if label[i] in cats:
            new_label[i] = newcat
    return new_dataset, new_label

def bayes():
    data = pd.read_csv('train.data.tsv', sep='\t', encoding='utf-8')
    contents = data['text'].values.tolist()
    label_data = data['label'].values.tolist()
    test_f = open("./textdata/test.data", encoding='utf-8')
    test_data = test_f.read().splitlines()
    test_data = [text.split('\t')[1] for text in test_data]
    texts = ['  '.join([word.lower() for word in jieba.cut(regex.sub(' ', str(text)), cut_all=False) if word.strip()]) for text in tqdm(contents, desc='traindata')]
    train_y = [int(label) for label in label_data]
    # model_2 = Word2Vec(size=300, min_count=1)
    # model_2.build_vocab(texts)
    # total_examples = model_2.corpus_count
    # model = KeyedVectors.load_word2vec_format("./pretrained/sgns.weibo.word", binary=False, unicode_errors='ignore')
    # model_2.build_vocab([list(model.vocab.keys())], update=True)
    # model_2.intersect_word2vec_format("./pretrained/sgns.weibo.word", binary=False, unicode_errors='ignore')
    # model_2.train(texts, total_examples=total_examples, epochs=model_2.iter)
    # vectors = []
    # for text in texts:
    #     vector = []
    #     for word in text:
    #         vector.append(model_2.wv.vectors[model_2.wv.vocab[word].index])
    #         if len(vector) >= 32:
    #             break
    #     while len(vector) < 32:
    #         vector.append(zeros)
    #     vectors.append(vector)
    # train_X = vectors
    # test_data = ['  '.join([word.lower() for word in jieba.cut(regex.sub(' ', str(text)), cut_all=False) if word.strip()]) for text in tqdm(test_data, desc='testdata')]
    # y_train = train_y
    X_train, X_test, y_train, y_test = train_test_split(texts, train_y, test_size=0.10, random_state=10)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
    X_full = tfidf.fit_transform(texts)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    clf = VotingClassifier(models, voting='soft', weights=[3, 2, 1])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(classification_report(y_test, y_predict))
    # with open('./submission.csv', 'w', encoding='utf-8') as f:
    #     f.writelines('ID,Expected\n')
    #     index = 0
    #     for i in y_predict:
    #         f.writelines(str(index) + ',' + str(i) + '\n')
    #         index += 1
#

if __name__ == '__main__':
    bayes()