import base64
import time

import mojimoji
import collections
from sklearn.preprocessing import StandardScaler
import numpy as np
import MeCab
import streamlit as st


#時間差を[hour]単位に変換
def timedelta_to_H(td):
    sec = td.total_seconds()
    return round(sec//3600, 1)

#勤務時間を開始時間と終了時間に分割する
def take_time(df):
    df["期間・時間\u3000勤務時間"] = df["期間・時間\u3000勤務時間"].apply(mojimoji.zen_to_han)
    df["期間・時間\u3000勤務時間"] = df["期間・時間\u3000勤務時間"].str.split(' ', expand=True)[0]
    df["期間・時間\u3000勤務時間start"] = df["期間・時間\u3000勤務時間"].str.split("〜", expand=True)[0]
    df["期間・時間\u3000勤務時間end"] = df["期間・時間\u3000勤務時間"].str.split("〜", expand=True)[1]
    df = df.drop(["期間・時間\u3000勤務時間"], axis=1)

#ターゲットエンコーディング
def target_encording(df, col_name):
    target_dict = df[[col_name, "応募者数 合計"]].groupby([col_name])["応募者数 合計"].mean().to_dict()
    encoded = df[col_name].map(lambda x: target_dict[x]).values
    df[f"{col_name}_応募者数 合計"] = encoded

#ラベルエンコーディング
def label_count_encording(df, col_name):
    counter = collections.Counter(df[col_name].values)
    count_dict = dict(counter.most_common())
    label_count_dict = {key: i for i, key in enumerate(count_dict.keys(), start=1)}
    encoded = df[col_name].map(lambda x: label_count_dict[x]).values
    df[f"{col_name}_ラベルカウント"] = encoded

#カウントエンコーディング
def count_encording(df, col_name):
    counter = collections.Counter(df[col_name].values)
    count_dict = dict(counter.most_common())
    encoded = df[col_name].map(lambda x: count_dict[x]).values
    df[f"{col_name}_カウント"] = encoded

#標準化を行う．log_Flagで対数変換を行うかどうかを選択
def standard_col(df, col_name, log_Flag=True):
    data = df[col_name].values
    if log_Flag:
        data = np.log(data + 1)

    standard = StandardScaler()
    norm = standard.fit_transform(data.reshape(-1, 1))
    df[col_name] = norm

#テキストをわかち書き化する
def wakati(text):
    tagger = MeCab.Tagger('')
    tagger.parse('')
    node = tagger.parseToNode(text)
    word_list = []
    while node:
        pos = node.feature.split(",")[0]
        if pos in ["名詞", "形容詞"]:   # 対象とする品詞
            word = node.surface
            word_list.append(word)
        node = node.next
    return " ".join(word_list)

#テキスト中の大文字を小文字にする
def lower_text(text):
    return text.lower()

#予測結果のCSVファイルをダウンロードさせるための関数
def csv_downloader(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_filename = "{}_test_pred.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">予測データをダウンロード</a>'
    st.markdown(href, unsafe_allow_html=True)
