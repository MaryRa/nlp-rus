import joblib
import re
import os

import pandas as pd
import numpy as np
import pymorphy2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class BadUsersBehavior:
    def __init__(self, n_jobs=1):
        self._n_jobs = n_jobs

        self.dict_replace = {'<NUMBER>': ['NUMR', 'NUMB'], '<LATN>': ['LATN']}
        self.num_list = {'один': '<NUMBER>', 'ноль': '<NUMBER>',
                         'пятёрка': '<NUMBER>'}
        signal_words = ['telegram', 'телеграм', 'телега',
                        'intagram', 'инста', 'инстаграм', 'инстограм', 'inst',
                        'viber', 'вайбер', 'вибер',
                        'whatsapp', 'вотсап', 'ватсап',
                        'mail', 'почта', 'мыло',
                        'телефон', 'тел',
                        'ютуб', 'youtube',
                        'www', 'http', 'https', 'ru', 'com', '@mail', '@gmail',
                        'discord',
                        '<NUMBER>', '<POTENTIALTELE>', '<ENDTEXT>',
                        '<STARTTEXT>']
        self.word_dict = dict(zip(signal_words, signal_words))

        self.morph = pymorphy2.MorphAnalyzer()
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2),
                                          min_df=2, max_df=0.90,
                                          dtype=np.float32)
        self.model = LogisticRegression(max_iter=1000, C=1.0,
                                        n_jobs=self._n_jobs)

    @staticmethod
    def _simple_clear(txt_series):
        return txt_series.str.lower().str.replace('ё', 'е')

    @staticmethod
    def _get_only_words_and_numb(txt):
        txt = re.sub("[^a-zа-я0-9@Ց]", " ", txt)
        return txt

    @staticmethod
    def _repl_potential_tele(txt):
        txt1 = re.sub(
            "(?:[ ]|^)[78]?[ ]?[94][\dՑоolisdbзв]{2}[ ]?[\dՑоolisdbзв]{3}[ ]?[\dՑоolisdbзв]{2}[ ]?[\dՑоolisdbзв]{2}[ ]?(?:[ ]|$)",
            ' <NUMBER> ', txt)
        txt1 = re.sub("\d+", ' <NUMBER> ', txt1)
        txt1 = '<STARTTEXT> ' + txt1 + ' <ENDTEXT>'
        return txt1

    def _get_mask_latn_numr(self, x):
        if x in self.word_dict.keys():
            return self.word_dict[x]
        item = self.morph.parse(x)[0].normal_form
        if item in self.num_list.keys():
            self.word_dict[x] = self.num_list[item]
            return self.word_dict[x]
        item = self.morph.parse(x)[0].tag
        t = [y for y in self.dict_replace.keys() if
             any([yy in item for yy in self.dict_replace[y]])]
        if len(t) != 0:
            self.word_dict[x] = t[0]
            return t[0]
        self.word_dict[x] = x
        return x

    def _repl_with_mask_latn_numr(self, txt):
        txt2 = [self._get_mask_latn_numr(x) for x in txt.split()]
        return ' '.join(txt2)

    def _preprocessing(self, df):
        df['description'] = self._simple_clear(df['description'])
        df['description_clear'] = df['description'].map(
            self._get_only_words_and_numb)
        df['description_clear'] = df['description_clear'].map(
            self._repl_potential_tele)
        df['description_clear'] = df['description_clear'].map(
            self._repl_with_mask_latn_numr)
        return df

    def fit(self, train, y_name='is_bad'):
        train_x = train.drop(columns=y_name)
        train_y = train[y_name]

        train_x = self._preprocessing(train_x)
        train_x_trans = self.vectorizer.fit_transform(
            train_x['description_clear'])
        self.model.fit(train_x_trans, train_y)
        print("Модели обучены")

    def load_models(self, path="./"):
        self.word_dict = joblib.load(path + "vocab.json")
        self.vectorizer = joblib.load(path + "vectorizer.sav")
        self.model = joblib.load(path + "model.sav")
        print("Модели скачаны")

    def save_models(self, path="./"):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.vectorizer, path + "vocab.json", compress=3)
        joblib.dump(self.vectorizer, path + "vectorizer.sav", compress=3)
        joblib.dump(self.model, path + "model.sav", compress=3)
        print("Модели сохранены")

    def predict_proba(self, test):
        test_x = self._preprocessing(test)
        test_x_trans = self.vectorizer.transform(test_x['description_clear'])
        test_x['prediction'] = self.model.predict_proba(test_x_trans)[:,
                               self.model.classes_ == 1]

        test_x['start'] = None
        test_x['end'] = None
        ind_more_05 = (test_x['prediction'] > 0.5)
        test_x[ind_more_05] = self._add_start_end_symbols(test_x[ind_more_05],
                                                          test_x_trans[
                                                          ind_more_05.values,
                                                          :])
        return test_x

    def predict(self, test, step=0.5):
        test_x = self.predict_proba(test)
        test_x['prediction'] = (test_x['prediction'] >= step) * 1
        return test_x

    @staticmethod
    def av_roc_auc(df, y_name, y_name_pred):
        by_cat = {x: roc_auc_score(df.loc[df['category'] == x, y_name],
                                   df.loc[df['category'] == x, y_name_pred])
                  for x in df['category'].unique()}
        return by_cat, np.mean(list(by_cat.values()))

    def _add_start_end_symbols(self, df, df_trans_matrix_vec):
        col = df.columns
        coefs = self.model.coef_.ravel()
        names = np.array(self.vectorizer.get_feature_names_out())
        coefs_mult_matrix = df_trans_matrix_vec.multiply(coefs)
        argmax_coefs = np.argmax(coefs_mult_matrix, axis=1)

        df['template'] = [self._define_template(names[x[0, 0]]) for x in
                          argmax_coefs]
        df['descr_clear_low'] = df['description_clear'].str.replace('<',
                                                                    "").str.replace(
            '>', "").str.lower()
        df[['start', 'end']] = df.apply(
            lambda x: self._get_index_symbols(x['template'],
                                              x['description_clear'],
                                              x['description'],
                                              x['descr_clear_low']),
            axis=1).replace(-1, "")
        return df[col]

    @staticmethod
    def _define_template(sign):
        before = "[\w]* ?(?:latn |number |starttext |^)+"
        after = "(?: latn| number| endtext|$)+ ?[\w]*"
        return f'(?:{before}{sign}{after}|{before}{sign}|{sign}{after})'

    def _get_index_symbols(self, template, descr_clear, descr,
                           descr_clear_low):
        last_symbol = -1
        first_symbol = -1

        samples = re.findall(template, descr_clear_low)
        if len(samples) == 1 and any(
                [x in samples[0] for x in ['number', 'latn']]):
            sample = samples[0].split()
            base = " *".join([sample[i] for i in range(len(sample)) if
                              i == 0 or sample[i - 1] != sample[i]])
            base = base.replace('number', '[а-я0-9 ]+').replace('endtext',
                                                                '$').replace(
                'starttext', '^').replace('latn', '[a-z ]+')
            part_from_descr = re.findall(base, descr)
            if len(part_from_descr) == 1:
                part_from_descr = part_from_descr[0]
                first_symbol = descr.find(part_from_descr)
                dict_fill = {x: (
                    x if x not in descr_clear_low.split() else ' ' * len(x))
                             for x in part_from_descr.split()}
                for x in dict_fill.keys():
                    part_from_descr = part_from_descr.replace(x, dict_fill[x])
                first_symbol += len(part_from_descr) - len(
                    part_from_descr.lstrip())
                last_symbol = first_symbol + len(part_from_descr.strip())
                if len(descr[first_symbol:last_symbol].strip()) == 0:
                    first_symbol = -1
                    last_symbol = -1
        return pd.Series({"start": first_symbol, "end": last_symbol})