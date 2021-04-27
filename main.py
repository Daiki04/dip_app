import streamlit as st
import pandas as pd
from datetime import datetime as dt
import mojimoji
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import collections
from sklearn.preprocessing import StandardScaler
import numpy as np
import MeCab
import base64
import time

import sub

st.title("""
応募者数を予測するWebアプリ
""")

st.sidebar.write("""
    ## 予測を始めよう！
""")

#ファイルアップロード部分
uploaded_file = st.sidebar.file_uploader("予測対象のCSVファイルをアップロードして下さい", ["csv"], accept_multiple_files=False)
load_text = st.empty()
size_text = st.empty()
if uploaded_file is not None:
    try:
        with st.spinner("データ読み込み中．．．"):
            data_up = pd.read_csv(uploaded_file)
            st.dataframe(data_up.head(11))
            load_text.markdown("**予測対象データ**")
            size_text.write(f"{data_up.shape[0]}行{data_up.shape[1]}列")
    except:
        st.sidebar.error("CSV形式のファイルを入力して下さい")

    #予測部分
    end_text = st.sidebar.empty()
    start_button = st.sidebar.empty()
    if start_button.button("予測開始"):
        start_button.empty()
        with st.spinner("予測中．．．"):
            #予測に用いる列の選択
            try:
                test_col_name = pd.read_csv("./data/num_columns.csv")
                test_col_name = test_col_name["列名"]
                test = data_up[test_col_name]
            except:
                st.error("データの入力フォーマットが正しくありません")

            #欠損値の穴埋め
            null_cols = test.isnull().sum().reset_index()
            null_cols = null_cols.rename(columns={"index": "列名", 0: "counts"})
            null_cols_over_one = null_cols[null_cols["counts"] > 1]

            value_col = ["（派遣先）配属先部署　人数", "勤務地　最寄駅1（分）",
                         "（派遣先）配属先部署　男女比　女", "給与/交通費　給与下限"]

            for col_name in null_cols_over_one["列名"]:
                if col_name in value_col:
                    test[col_name] = test[col_name].fillna(round(test[col_name].mean(), 0))
                else:
                    test[col_name] = test[col_name].fillna(test[col_name].mode()[0])

            #勤務時間（時間単位）の取得
            sub.take_time(test)
            for i in range(test.shape[0]):
                test["期間・時間\u3000勤務時間start"][i] = dt.strptime(test["期間・時間\u3000勤務時間start"][i], '%H:%M')
                test["期間・時間\u3000勤務時間end"][i] = dt.strptime(test["期間・時間\u3000勤務時間end"][i], '%H:%M')
            test["期間・時間\u3000勤務時間delta"] = (test["期間・時間\u3000勤務時間end"] - test["期間・時間\u3000勤務時間start"])
            for i in range(test.shape[0]):
                test["期間・時間\u3000勤務時間delta"][i] = sub.timedelta_to_H(test["期間・時間\u3000勤務時間delta"][i])

            #拠点番号の番号部分の取得
            test["拠点番号"] = test["拠点番号"].str[:3]
            #カテゴリー変数のエンコーディング
            sub.count_encording(test, "お仕事No.")
            sub.label_count_encording(test, "お仕事No.")
            col_names = ["職種コード", "フラグオプション選択", "勤務地　備考", "拠点番号", "勤務地　最寄駅1（沿線名）", "会社概要　業界コード", "勤務地　都道府県コード", "仕事の仕方", "勤務地　市区町村コード", "勤務地　最寄駅1（駅名）", "期間・時間　勤務時間start", "期間・時間　勤務時間end", "期間・時間　勤務時間delta"]

            for col_name in col_names:
                sub.count_encording(test, col_name)
            for col_name in col_names:
                sub.label_count_encording(test, col_name)

            test = test.drop(["職種コード", "勤務地　備考", "拠点番号", "勤務地　最寄駅1（沿線名）",
                                "会社概要　業界コード", "勤務地　都道府県コード", "仕事の仕方", "勤務地　市区町村コード", "勤務地　最寄駅1（駅名）"], axis=1)

            #数値データの標準化
            test["職場の様子"] = pd.Categorical(test["職場の様子"], categories=[2, 3, 4])
            test["フラグオプション選択"] = pd.Categorical(test["フラグオプション選択"], categories=[0, 1, 2, 3, 5])
            test["勤務地　最寄駅1（駅からの交通手段）"] = pd.Categorical(test["勤務地　最寄駅1（駅からの交通手段）"], categories=[1.0, 2.0, 3.0])
            test["給与/交通費　交通費"] = pd.Categorical(test["給与/交通費　交通費"], categories=[2, 3])
            test = pd.get_dummies(test, columns=["職場の様子", "フラグオプション選択", "勤務地　最寄駅1（駅からの交通手段）", "給与/交通費　交通費"])

            counter = collections.Counter(test['給与/交通費\u3000給与下限'].values)
            count_dict = dict(counter.most_common())
            label_count_dict = {key:i for i, key in enumerate(count_dict.keys(), start=1)}
            encoded = test['給与/交通費\u3000給与下限'].map(lambda x: label_count_dict[x]).values
            test['給与/交通費\u3000給与下限_rank'] = encoded

            st_colnames = ["給与/交通費\u3000給与下限", "（派遣先）配属先部署　人数",
                          "（派遣先）配属先部署　男女比　女", "勤務地\u3000最寄駅1（分）"]

            for col_name in st_colnames:
                sub.standard_col(test, col_name)

            #お仕事No.の重複をなくす
            job_nums = test["お仕事No."].value_counts().reset_index()
            job_nums = job_nums.rename(columns={"index": "お仕事No.", "お仕事No.": "counts"})
            job_nums_over_one = job_nums[job_nums["counts"] > 1]

            test = test.drop_duplicates(subset="お仕事No.").reset_index(drop=True)

            #テキストデータから応募者数期待度を予測する
            text_col = pd.read_csv("./data/text_col_name.csv")
            text_col = text_col["0"]
            test_text = data_up[text_col]

            for column in text_col:
                test_text[column] = test_text[column].apply(mojimoji.zen_to_han)
            for column in text_col:
                test_text[column] = test_text[column].apply(sub.wakati)

            for column in test_text.columns:
                test_text[column] = test_text[column].apply(sub.lower_text)

            file_name = "./data/tfidf_vect.pkl"
            vectorizer = None
            with open(file_name, 'rb') as f:
                vectorizer = pickle.load(f)

            test_tfidfs = pd.DataFrame()

            for column in text_col:
                tsne = TSNE(n_components=2, verbose=1, n_iter=500)
                test_tfidf = vectorizer.transform(test_text[column])
                test_df = pd.DataFrame(test_tfidf.toarray(), columns=vectorizer.get_feature_names())
                test_tsne = tsne.fit_transform(test_df)
                test_tsne_df = pd.DataFrame(test_tsne, columns=[f"{column}x", f"{column}y"])
                test_tfidfs = pd.concat([test_tfidfs, test_tsne_df], axis=1)

            filename = "./data/xgbc.sav"
            bst = pickle.load(open(filename, 'rb'))

            dtest = xgb.DMatrix(test_tfidfs)
            test_pred = bst.predict(dtest)

            test["応募者数期待度"] = test_pred

            #整形したデータによる予測
            X = test.drop(["お仕事No.", "期間・時間　勤務時間start", "期間・時間　勤務時間end", "期間・時間　勤務時間delta", "期間・時間　勤務時間"], axis=1)

            filename = "./data/xgbr_best.sav"
            xgbr = pickle.load(open(filename, 'rb'))
            test_dataset = xgb.DMatrix(X)
            test_pred = xgbr.predict(test_dataset)

            sub_test = pd.DataFrame(test["お仕事No."])
            sub_test["応募数 合計"] = test_pred
        st.balloons()
        end_text.markdown("**予測完了**")
        st.markdown("**予測結果**")
        st.write(sub_test)
        sub.csv_downloader(sub_test)
        start_button.button("再実行")