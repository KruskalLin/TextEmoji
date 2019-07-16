from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re

regex = re.compile(r'[u4e00-u9fa5A-Za-z！？~～。….?!#$￥]')
jieba.load_userdict('./textdata/dict.txt')


def train():
    data = pd.read_csv('train.data.tsv', sep='\t', encoding='utf-8', converters={'label': np.int, 'text': str})

    y = data["label"].values.tolist()
    # test_f = open("./textdata/test.data", encoding='utf-8')
    # test_data = test_f.read().splitlines()
    # test_f.close()
    # test_data = pd.DataFrame(test_data, columns=['text'])
    # test_data["split_words"] = test_data["text"].apply(lambda x: [word.lower() for word in jieba.cut(regex.sub(' ', str(x)), cut_all=False) if word.strip()])

    # data["split_words"] = data["text"].apply(lambda x: [word.lower() for word in jieba.cut(regex.sub(' ', str(x)), cut_all=False) if word.strip()])
    # tfidf = TfidfVectorizer()
    # train_texts = data["split_words"].apply(lambda x: ' '.join(x)).values.tolist()

    # test_texts = test_data["split_words"].apply(lambda x: ' '.join(x)).values.tolist()
    # full_tfidf = tfidf.fit_transform(train_texts)
    # train_tfidf = tfidf.transform(train_texts)
    # test_tfidf = tfidf.transform(test_texts)

    del data
    train_data = pd.read_csv('train_data.tsv', sep='\t', encoding='utf-8')
    # test_data = pd.read_csv('test_data.tsv', sep='\t', encoding='utf-8')

    # print(data.info())
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.10, random_state=100)
    # del y
    # del data
    params = {
        'boosting_type': 'dart',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'drop_rate': 0.1,
        'n_estimators': 150,
        'num_class': 72,
        'learning_rate': 0.1,
        'is_unbalance': 'true',
        'num_leaves': 127,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 500,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.9,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 1.,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.2,  # L1 regularization term on weights
        'reg_lambda': 1.,  # L2 regularization term on weights
        'verbosity': 0,
        'device': 'gpu',
        'gpu_use_dp': 'true'
    }
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    print(classification_report(y_test, y_predict))
    # with open('./submission.csv', 'w', encoding='utf-8') as f:
    #     f.writelines('ID,Expected\n')
    #     index = 0
    #     for i in y_predict:
    #         f.writelines(str(index) + ',' + str(i) + '\n')
    #         index += 1

if __name__ == '__main__':
    train()