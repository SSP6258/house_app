import time
import pandas as pd
import numpy as np
import os
import datetime
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import xgboost as xgb
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.mixture import GaussianMixture
from plotly.subplots import make_subplots
from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall
from st_aggrid import AgGrid
from PIL import Image
from collections import defaultdict
from dataprep.eda import plot_correlation
from house_utils import fn_get_geo_info, fn_get_admin_dist, dic_of_path, geodesic
from house_elt import fn_addr_handle, fn_house_coor_read, fn_house_coor_save
from house_elt import fn_gen_build_case, fn_gen_house_data

# pip list --format=freeze > requirements.txt

dic_of_cn_2_en = {'經度': 'longitude',
                  '緯度': 'latitude',
                  '移轉層次': 'Floor',
                  '車位類別_其他': 'p_other',
                  '車位類別_坡道機械': 'p_ramp_machine',
                  '車位類別_一樓平面': 'p_1f_plane',
                  '車位類別_坡道平面': 'p_ramp_plane',
                  '車位類別_升降機械': 'p_lift_machine',
                  '車位類別_塔式車位': 'p_tower',
                  '車位類別_升降平面': 'p_lift_plane',
                  '交易年': 'trade_year',
                  '總樓層數': 'Floors',
                  '頂樓': 'roof',
                  '頂樓-1': 'roof-1',
                  '台北市': 'TPE',
                  '屋齡': 'Age',
                  '主要建材_RC': 'RC',
                  '主要建材_SRC': 'SRC',
                  '主要建材_SC': 'SC',
                  '建物坪數': 'building acreage',
                  '車位坪數': 'parking acreage',
                  '幾廳': 'livingrooms',
                  '幾衛': 'bathrooms',
                  '幾房': 'rooms',
                  '利率_15個月前': 'interest rate(15M ago)',
                  '利率_13個月前': 'interest rate(13M ago)',
                  '使用分區_住': 'land_typ'
                  }


def fn_show_img(IMG_path, IMG_file):
    png = os.path.join(IMG_path, IMG_file)
    img = Image.open(png)
    st.image(img)


def fn_addr_2_house_num(x):
    num = x.split('路')[-1]
    num = num.split('街')[-1]
    num = num.split('段')[-1]
    num = num.split('巷')[-1]
    num = num.split('弄')[-1]

    if num == x:
        print(x, 'special addr !!!')

    return num


def fn_addr_2_build_case(addr):
    build_case = 'No_build _case_found'
    build_case_str = 'NA'
    df_coor_read = fn_house_coor_read()
    addr = fn_addr_handle(addr)

    if addr in df_coor_read.index:
        build_case_str = str(df_coor_read.loc[addr, 'Build case'])

    if build_case == 'No_build_case_found':
        geo, is_save, is_match, addr_fr_db = fn_get_geo_info(addr, df_coor_read)
        lat = geo['coor']['lat']
        lon = geo['coor']['log']
        print(lat, lon, addr)
        df_sel = df_coor_read[df_coor_read['lat'].apply(lambda x: abs(x - lat) < 0.00001)]
        df_sel = df_sel[df_sel['lon'].apply(lambda x: abs(x - lon) < 0.00001)]
        df_sel = df_sel[df_sel['Build case'].apply(lambda x: str(x) not in ['nan', 'NA'])]
        if df_sel.shape[0]:
            builds = df_sel['Build case'].unique()
            build_case_str = builds[0]
            if len(builds) > 1:
                print(df_sel['Build case'].unique())

    if build_case_str != 'nan' and build_case_str != 'NA' and not build_case_str.endswith('區'):
        build_case = build_case_str
        print(addr, '-->', build_case)

    return build_case


# Anomaly Detection using Gaussian Mixtures
def fn_anomaly_detection(df, n_comp, percent):
    df_det = df[['MRT_DIST', '總樓層數', '移轉層次', 'lat', 'log', '每坪單價(萬)']]

    n_comp = int(df.shape[0] / 2) if df.shape[0] < n_comp else n_comp

    gm = GaussianMixture(n_components=n_comp, n_init=10, random_state=42)
    gm.fit(df_det)
    densities = gm.score_samples(df_det)
    density_threshold = np.percentile(densities, percent)
    anomalies = df_det[densities < density_threshold]
    df['ano'] = densities < density_threshold

    return df


def fn_cln_house_data(df):
    df['city'] = df['土地位置建物門牌'].apply(lambda x: x.split('市')[0].replace('臺', '台') + '市')
    df['建物移轉坪數'] = df['建物移轉坪數'].apply(lambda x: round(x, 2))
    df['建物型態'] = df['建物型態'].apply(lambda x: '華廈' if '華廈' in x else '大樓' if '大樓' in x else x)
    df.rename(columns={col: col.replace('移轉坪數', '坪數') for col in df.columns}, inplace=True)
    df = df[df['車位總價元'].astype(float) > 0] if '車位總價元' in df.columns else df
    df = fn_gen_build_case(df)
    df[['經度', '緯度']] = df[['log', 'lat']]

    return df


@st.cache(allow_output_mutation=True)
def fn_load_model(model_sel):
    try:
        loaded_model = pickle.load(open(model_sel, 'rb'))
    except:
        assert False, f'fn_load_model() fail, from {model_sel}'

    return loaded_model


def fn_set_radio_2_hor():
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


def fn_set_color_by(by, df):
    color_set = None
    opacity = 0.3

    if by == '依捷運距離':
        color_set = df['MRT_DIST']
    elif by == '依通勤時間':
        color_set = df['MRT_Commute_Time_UL']
    elif by == '依交易年':
        color_set = df['交易年']
    elif by == '依小學距離':
        color_set = df['sku_dist']
    elif by == '依小學人數':
        color_set = df['sku_109_total']
    elif by == '依總樓層數':
        color_set = df['總樓層數']
    elif by == '依建物坪數':
        color_set = df['建物坪數']
    elif '依最新登錄' in by:
        color_set = df['Latest']
        opacity = 0.6
    elif '依行政區' in by:
        label_encoder = LabelEncoder()
        color_set = label_encoder.fit_transform(df['鄉鎮市區'])

    return color_set, opacity


@st.cache
def fn_get_house_data(path):
    df = pd.read_csv(path)
    read_typ = path.replace('\\', '/').split('/')[-3]
    is_merge_pre_own = False

    if read_typ == 'pre_sold_house' and is_merge_pre_own:
        pre_ownd_path = path.replace('pre_sold_house', 'pre_owned house')
        df.drop(columns=['棟及號'], inplace=True)
        if os.path.exists(pre_ownd_path):
            df_ownd = pd.read_csv(pre_ownd_path)
            df_ownd = df_ownd[df_ownd['屋齡'] == 0]

            df_ownd['建築完成年月'] = df_ownd['建築完成年月'].apply(lambda x: np.nan)
            if df_ownd.shape[0]:
                col_sold = df.columns
                col_ownd = df_ownd.columns
                col_drop = []
                for c in col_ownd:
                    if c not in col_sold:
                        col_drop.append(c)
                df_ownd.drop(columns=col_drop, inplace=True)
                df = df.append(df_ownd)
                # df.drop_duplicates(subset=['地址', '交易年月日', '總樓層數', '移轉層次', '每坪單價(萬)', '建物移轉坪數'],
                #                    inplace=True)
                # df.reset_index(drop=True, inplace=True)
                df.to_csv(path.replace('.csv', f'_add_{df_ownd.shape[0]}.csv'), encoding='utf-8-sig', index=False)
                print(f'Append {df_ownd.shape[0]} data from pre_ownd to {read_typ} and total is {df.shape[0]}')

    df.drop_duplicates(subset=['地址', '交易年月日', '總樓層數', '移轉層次', '每坪單價(萬)', '建物移轉坪數'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'Read {read_typ} data from {path} !!!')
    return df


def fn_get_neighbor(target, neighbors):
    neighbors = list(neighbors)
    path = dic_of_path['database']
    path_db = os.path.join(path, 'School_info.csv')
    df = pd.read_csv(path_db, encoding='utf-8-sig')
    coor_t = tuple(df[df['schoolname'] == target][['lat', 'lon']].values[0])

    distances = []
    for n in neighbors:
        coor_n = tuple(df[df['schoolname'] == n][['lat', 'lon']].values[0])
        distances.append(int(geodesic(coor_t, coor_n).meters))

    closest = neighbors[distances.index(min(distances))]

    st.write(f'鄰近小學: {target}, 均價參考: {closest}')

    return closest


def fn_get_sku_people_by_year(df):
    path = dic_of_path['database']
    file = os.path.join(path, 'School_info.csv')
    columns = df.columns
    if '交易年' in columns and 'sku_name' in columns:
        df_sku = pd.read_csv(file, encoding='utf-8-sig')
        for idx in df.index:
            year = df.loc[idx, '交易年'] - 1
            school = df.loc[idx, 'sku_name']
            city = '台北市' if df.loc[idx, '台北市'] else '新北市'

            df_sku_sel = df_sku[df_sku['schoolname'] == school]
            df_sku_sel = df_sku_sel[df_sku_sel['city'] == city]
            assert df_sku_sel.shape[0] == 1, f'{city, school, df_sku_sel.shape[0]}'

            for y in range(10):
                year_total = f'{y + year}_Total'
                if year_total in df_sku_sel.columns:
                    total = df_sku_sel[year_total].values[0]

                if str(total) == 'nan':
                    if y:
                        print(f'can NOT find sku total of {city, school, total, year} try {year_total}')
                else:
                    break

            assert str(total) != 'nan', f'{idx, total, city, school, year_total, year}'
            df.at[idx, 'sku_total'] = total
    else:
        print(columns)

    return df


def fn_get_interest_rate(df, months=1):
    path = dic_of_path['database']
    file = os.path.join(path, 'a13rate.csv')
    df['交易年月日'] = df['交易年'] * 10000 + 900 if '交易年月日' not in df.columns else df['交易年月日']

    df_rate = pd.read_csv(file, encoding='utf-8-sig', header=4)
    rate_col = df_rate.columns[13]  # 定存利率
    date_col = df_rate.columns[0]
    for idx in df.index:
        trade_date = float(int(df.loc[idx, '交易年月日'] / 100))

        if trade_date in df_rate[date_col].values:
            df_t = df_rate[df_rate[date_col] <= trade_date]
            rates = df_t[rate_col].values
            rate_sel = []
            for m in range(months):
                try:
                    rate_sel.append(rates[-1 - m])
                except:
                    rate_sel.append(rate_sel[-1])

                df.at[idx, f'利率_{m}個月前'] = rate_sel[-1]
        else:
            assert False, f'can NOT find interest_rate of {trade_date}'

    return df


def fn_get_categories(path, feature):
    df_f = pd.read_csv(os.path.join(path, f'output\\Feature_{feature}.csv'))
    cats = df_f[feature].to_list()

    return cats


def fn_get_hover_text(df):
    txt = ''
    cols = df.columns

    if '鄉鎮市區' in cols:
        txt += df['鄉鎮市區']

    if '建案名稱' in cols:
        txt += df['建案名稱'].astype(str)

    if '交易年' in cols:
        txt += df['交易年'].astype(str) + '年, '

    if 'MRT DIST' in cols:
        txt += df['MRT_DIST'].astype(int).astype(str) + '公尺, '

    if 'MRT_Commute_Time_UL' in cols:
        txt += df['MRT_Commute_Time_UL'].astype(str) + '分鐘'

    return txt


def fn_gen_pred(path, model, model_name, df_F, build_typ, is_rf):
    st.write('')
    st.subheader('批次驗證')
    st.write("驗證資料:[内政部不動產成交案件 資料供應系統(每月1、11、21日發布)](https://plvr.land.moi.gov.tw/DownloadOpenData)")

    file = st.file_uploader("資料上傳", type=['csv'])
    print(file)
    if not file:
        st.write(' please upload *.csv to test')
    else:
        df = pd.read_csv(file)
        ave_path = dic_of_path['database']
        df_sku_ave = pd.read_csv(os.path.join(ave_path, 'SKU_ave.csv'), index_col='sku_name')
        df_mrt_ave = pd.read_csv(os.path.join(ave_path, 'MRT_ave.csv'), index_col='MRT')
        df_dist_ave = pd.read_csv(os.path.join(ave_path, 'DIST_ave.csv'), index_col='鄉鎮市區')

        n_data = df.shape[0] - 1
        temp = os.path.join(path, 'output\\temp')
        if not os.path.exists(temp):
            os.makedirs(temp)
        df = fn_gen_house_data(os.path.join(temp, file.name), 'test', df_validate=df)

        df['MRT_ave'] = df['MRT'].apply(lambda x: df_mrt_ave.loc[x, '每坪單價(萬)'])
        df['SKU_ave'] = df['sku_name'].apply(lambda x: df_sku_ave.loc[x, '每坪單價(萬)'])
        df['DIST_ave'] = df['鄉鎮市區'].apply(lambda x: df_dist_ave.loc[x, '每坪單價(萬)'])

        if df.shape[0] and n_data:

            df = fn_cln_house_data(df.copy())

            df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df
            for i in df['主要建材'].tolist():
                if i not in df['主要建材'].unique():
                    print(i)

            X, df_cat = fn_gen_training_data(df.copy(), path, is_inference=True, df_F=df_F)
            df['模型預估(萬/坪)'] = model.predict(X)

            if is_rf:
                trees, conf = fn_gen_model_confidence(model, X)
                df['信心指標'] = conf

            df['模型預估(萬/坪)'] = df['模型預估(萬/坪)'].apply(lambda x: round(x, 2))
            df['差(萬/坪)'] = df['模型預估(萬/坪)'] - df['每坪單價(萬)']
            df['誤差(萬/坪)'] = df['差(萬/坪)'].apply(lambda x: round(x, 2))
            df = df[df['移轉層次'] > 1]
            df.sort_values(by=['誤差(萬/坪)'], inplace=True, ignore_index=True)
            n_test = df.shape[0]
            st.write(f'此檔共有{n_data}筆 資料, 經篩選後有 {n_test}筆 可進行模型預估')

            show_cols = ['信心指標'] if is_rf else []
            show_cols += ['誤差(萬/坪)', '每坪單價(萬)', '模型預估(萬/坪)', '地址', '移轉層次', 'MRT']
            df_show = df[show_cols]

            # st.dataframe(df_show)

            if is_rf:
                config = {'scrollZoom': True,
                          'toImageButtonOptions': {'height': None, 'width': None}}

                st.write('')
                st.subheader(f'模型可信度分析')
                c1, c2 = st.columns(2)
                ths = c2.slider('信心門檻', min_value=90, max_value=100, value=(96, 100))
                th_l, th_h = ths[0], ths[1]
                df_sel = df[df['信心指標'].apply(lambda x: th_h >= x >= th_l)]

                colors = ['無', f'依行政區({len(df["鄉鎮市區"].unique())})', '依捷運距離', '依通勤時間']
                color_by = c1.selectbox('著色條件:', colors)

                margin = dict(t=50, l=20, r=0, b=0)
                fig = make_subplots()
                color_set, opacity = fn_set_color_by(color_by, df)
                hover_text = fn_get_hover_text(df)
                fig = fn_gen_plotly_scatter(fig, df['每坪單價(萬)'], df['模型預估(萬/坪)'], margin=margin,
                                            color=color_set, text=hover_text, opacity=0.6)
                color_set, opacity = fn_set_color_by(color_by, df_sel)
                hover_text = fn_get_hover_text(df_sel)
                title = f'模型: ml_model{model_name.split("ml_model")[-1]} 的 可信度評估 <br>' \
                        f'( 此模型進行{df.shape[0]}筆預測, 信心指標介於 {th_l} ~ {th_h} ' \
                        f'的 有{df_sel.shape[0]}筆, 約{int(100 * df_sel.shape[0] / df.shape[0])}% )'

                fig = fn_gen_plotly_scatter(fig, df_sel['每坪單價(萬)'], df_sel['模型預估(萬/坪)'], margin=margin,
                                            color=color_set, text=hover_text, opacity=1,
                                            xlabel='實際單價(萬/坪)', ylabel='預估單價(萬/坪)', title=title)
                st.write('')
                st.plotly_chart(fig, config=config)

                fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 1, "colspan": 2}, None], [{}, {}]],
                                    subplot_titles=('信心指標v.s.絕對誤差', '信心分佈', '誤差(萬/坪)分佈'))

                fig = fn_gen_plotly_hist(fig, df['信心指標'], '信心指標', row=2, col=1, margin=margin)
                fig = fn_gen_plotly_hist(fig, df['誤差(萬/坪)'], '誤差分布(萬/坪)', row=2, col=2, margin=margin)

                color_set, opacity = fn_set_color_by(color_by, df)

                # hover_text = df['鄉鎮市區']
                hover_text = fn_get_hover_text(df)

                fig = fn_gen_plotly_scatter(fig, df['信心指標'], abs(df['誤差(萬/坪)']), row=1, margin=margin,
                                            color=color_set, text=hover_text, opacity=0.6,
                                            xlabel='信心指標(分)', ylabel='絕對誤差(萬/坪)')
                # fig.add_vline(x=96, row=2, line dash="dash", line_color-"red")

                if df_sel.shape[0] > 0:
                    err_max = max(abs(df_sel['誤差(萬/坪)']))
                    fig.add_vrect(x0=-1 * err_max, row=2, col=2, x1=err_max, line_width=0, fillcolor="red", opacity=0.1)
                    fig.add_vrect(x0=th_l, row=2, col=1, x1=th_h, line_width=0, fillcolor="red", opacity=0.1)
                    fig.add_vrect(x0=th_l, row=1, col=1, x1=th_h, line_width=0, fillcolor="red", opacity=0.1)

                st.plotly_chart(fig, config=config)

            st.write('')
            AgGrid(df_show)

            del df
        else:
            st.write(f'此檔共有 {n_data}筆 資料, 經篩選後不可進行模型預估 !')


def fn_gen_training_data(df, path, is_inference=False, df_F=pd.DataFrame()):
    le = LabelEncoder()
    # df cat= df[['鄉鎮市區,主要建材',車位類別, 'MRT']]
    df.reset_index(drop=True, inplace=True)
    df_cat = df[['主要建材', '車位類別']]
    f_cat = []
    for col in df_cat.columns:
        cats = fn_get_categories(path, col)
        # df_f = pd.read_csv(os.path.join(path, f'output\\Feature_{col}.csv'))
        # cats = df_f[col].to_list()
        enc = OneHotEncoder(categories=[cats])
        one_hot = enc.fit_transform(df_cat[[col]]).toarray()
        df_oh = pd.DataFrame(one_hot)
        col_name = [col + '_' + str(cats[c]) for c in df_oh.columns]
        assert df.index[-1] == df_oh.index[
            -1], f'index NOT match !!! {df.shape, df_oh.shape} {df.index[-1], df_oh.index[-1]}'
        df[col_name] = df_oh
        df.drop(columns=col, inplace=True)
        f_cat = f_cat + col_name

    df['頂樓'] = df['移轉層次'].astype(int) - df['總樓層數'].astype(int)
    df['頂樓-1'] = df['頂樓'].apply(lambda x: 1 if x == -1 else 0)
    df['頂樓'] = df['頂樓'].apply(lambda x: 1 if x == 0 else 0)
    df['建物型態'] = df['總樓層數'].apply(lambda x: 1 if int(x) >= 11 else 0)

    df['使用分區_住'] = df['都市土地使用分區'].apply(lambda x: 1 if '住' in x else 0) if '都市土地使用分區' in df.columns else 1

    df = fn_get_sku_people_by_year(df.copy())
    df = fn_get_interest_rate(df.copy(), months=24)

    f_num = ['屋齡', '交易年']
    f_num += ['台北市', '緯度', '經度']
    f_num += ['建物坪數', '車位坪數', '幾房', '幾廳', '幾衛']
    f_num += ['總樓層數', '頂樓', '移轉層次']
    f_num += ['sku_dist', 'sku_total']
    f_num += ['MRT_DIST', 'MRT_Tput_UL', 'MRT_Tput_DL', 'MRT_Tput', 'MRT_Commute_Time_UL']
    f_num += ['利率_13個月前', '利率_15個月前']
    f_num += ['頂樓-1']
    f_num += ['使用分區_住']

    f_num += ['MRT_ave', 'SKU_ave', 'DIST_ave']

    if is_inference:
        Features = df_F['Features'].to_list()
    else:
        f_num += ['MRT']
        Features = f_num + f_cat

    if is_inference:
        Feature_sel = Features
    else:
        Feature_sel = []
        for f in Features:
            if len(df[f].unique()) > 1:
                Feature_sel.append(f)

    X = df[Feature_sel]

    for c in X.columns:
        if X[c].isna().any() and is_inference:
            print(c, X[[c]].shape, X.shape)
            for i, v in enumerate(X[c].tolist()):
                print(c, i, v, df_cat.iloc[i, :].values)
            assert False, c + str(f' is {X[c].values} {len(X[c].tolist())}')

    return X, df_cat


def fn_gen_model_explain(X, model):
    X.rename(columns={col: dic_of_cn_2_en[col] if col in dic_of_cn_2_en.keys() else col for col in X.columns},
             inplace=True)
    prediction, bias, contributions = treeinterpreter.predict(model, X.values)
    p = round(prediction[0][0], 2)
    b = round(bias[0], 2)
    cts = list(contributions[0])
    cts = [round(x, 2) for x in cts]
    fig = waterfall(X.columns, cts, threshold=0.01, sorted_value=True,
                    Title=f'Contribution of each Feature({X.shape[1]} features)\npred({p})= bias({b})+contributions({round(p - b, 2)})',
                    rotation_value=90, formatting='{:,.2f}')
    st.write('')
    st.subheader('模型解釋')
    st.pyplot(fig)
    print(X.columns)


def fn_gen_plotly_hist(fig, data, title, row=1, col=1, margin=None, bins=100, line_color='white', showlegend=False,
                       hovertext=None, barmode='group', opacity=0.8):
    fig.add_trace(
        go.Histogram(x=data, name=title, showlegend=showlegend, nbinsx=bins, hovertext=hovertext,
                     marker=dict(
                         opacity=opacity,
                         line=dict(
                             color=line_color, width=0.4
                         ),
                     )),
        row=row,
        col=col,
    )

    fig.update_layout(margin=margin,
                      barmode=barmode)

    return fig


def fn_gen_plotly_bar(df_top, x_data_col, y_data_col, text_col, v_or_h, margin,
                      color_col=None, text_fmt=None, title=None, ccs='agsunset'):
    fig = px.bar(df_top, x=x_data_col, y=y_data_col,
                 orientation=v_or_h, title=title,
                 text=text_col, color=color_col,
                 color_continuous_scale=ccs)

    fig.update_traces(texttemplate=text_fmt)
    fig.update_layout(margin=margin)

    return fig


def fn_gen_plotly_map(df, title, hover_name, hover_data, map_style,
                      color=None, zoom=10, height=400, text=None, margin=None):
    margin = {"r": 0, "t": 40, "l": 0, "b": 0} if margin is None else margin

    lat, lon = 'na', 'na'

    for y in ['緯度', 'lat']:
        if y in df.columns:
            lat = y
            break

    for x in ['經度', 'log', 'lon']:
        if x in df.columns:
            lon = x
            break

    assert lat != 'na' and lon != 'na', f'This df have no coor col: {df.columns}'

    fig = px.scatter_mapbox(df,
                            lat=lat, lon=lon, title=title,
                            hover_name=hover_name, hover_data=hover_data,
                            color_discrete_sequence=["fuchsia"],
                            zoom=zoom, height=height, color=color,
                            text=text)

    fig.update_layout(mapbox_style=map_style, margin=margin)
    # map style - "open-street-map", "white-bg", "carto-positron", "stamen-terrain"
    return fig


def fn_gen_plotly_scatter(fig, x_data, y_data, row=1, col=1, margin=None, color=None, text=None, opacity=0.3,
                          xlabel=None, ylabel=None, title=None):
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', showlegend=False,
                             marker=dict(
                                 opacity=opacity,
                                 line={'color': 'White', 'width': 0.4},
                                 color=color,
                                 colorscale='Bluered')  # "Viridis")
                             ), row=row, col=col)

    if margin is not None:
        fig.update_layout(margin=margin)

    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(size=13)
    )

    return fig


def fn_gen_df_color(val):
    color = 'greenyellow' if val else 'lightgray'

    return f'background-color: fcolor]'


def fn_gen_analysis_mrt(df, color_by, margin=None, bc_name=None):
    if bc_name is None:
        bc_name = ['康寶日出印象']
    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin
    mrts = len(df['MRT'].unique())

    fig_sct = make_subplots(rows=3, cols=2,
                            specs=[[{"colspan": 2, "rowspan": 2}, None], [{}, {}], [{}, {}]],
                            subplot_titles=(f'{mrts}個 鄰近捷運站 V.S. 每坪單價(萬)',))
    # df_sort = df.sort_values(by='每坪單價(萬),ascending=True)
    df_sort = df.sort_values(by='MRT_ave', ascending=False)
    df_hl = df_sort[df_sort['建案名稱'].apply(lambda x: x in bc_name)]

    hover_text = fn_get_hover_text(df_sort)

    color_set, opacity = fn_set_color_by(color_by, df_sort)

    fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['MRT'], df_sort['每坪單價(萬)'],
                                    margin=margin, color=color_set, text=hover_text)

    hover_txt1 = fn_get_hover_text(df_hl)

    fig_sct = fn_gen_plotly_scatter(fig_sct, df_hl['MRT'], df_hl['每坪單價(萬)'],
                                    margin=margin, color='red', text=hover_txt1, opacity=1)

    sub_titles = ['鄰近捷運通勤時間(分)', '鄰近捷運距離(公尺)', '上班時間進站人數', '上班時間出站人數']

    fig_sct_1 = make_subplots(rows=2, cols=2,
                              specs=[[{}, {}], [{}, {}]],
                              subplot_titles=('鄰近捷運通勤時間(分) v.s 每坪單價(萬)',
                                              '鄰近捷運距離(公尺) v.s 每坪單價(萬)',
                                              '上班時間進站人數 v.s 每坪單價(萬)',
                                              '上班時間出站人數 v.s 每坪單價(萬)'))

    # hover_text = fn_get_hover_text(df_sort)

    y_data = df_sort['每坪單價(萬)']
    y_hl = df_hl['每坪單價(萬)']
    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_sort['MRT_Commute_Time_UL'], y_data, row=1, col=1,
                                      margin=margin,
                                      color=color_set, text=hover_text)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_hl['MRT_Commute_Time_UL'], y_hl, row=1, col=1,
                                      margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_sort['MRT_DIST'], y_data, row=1, col=2, margin=margin,
                                      color=color_set, text=hover_text)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_hl['MRT_DIST'], y_hl, row=1, col=2, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_sort['MRT_Tput_UL'], y_data, row=2, col=1, margin=margin,
                                      color=color_set, text=hover_text)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_hl['MRT_Tput_UL'], y_hl, row=2, col=1, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_sort['MRT_Tput_DL'], y_data, row=2, col=2, margin=margin,
                                      color=color_set, text=hover_text)

    fig_sct_1 = fn_gen_plotly_scatter(fig_sct_1, df_hl['MRT_Tput_DL'], y_hl, row=2, col=2, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)
    return fig_sct, fig_sct_1


def fn_gen_analysis_sku(df, color_by, margin=None, bc_name=None):
    if bc_name is None:
        bc_name = ['康寶日出印象']
    SKUs = len(df['sku_name'].unique())
    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin

    df_sort = df.sort_values(by='SKU_ave', ascending=False)

    df_sort['sku_name'] = df_sort['sku_name'].apply(
        lambda x: x.replace('高中', '').replace('中學', '').replace('實驗', '').replace('國立', ''))

    df_hl = df_sort[df_sort['建案名稱'].apply(lambda x: x in bc_name)]
    color_set, opacity = fn_set_color_by(color_by, df_sort)

    y_data = df_sort['每坪單價(萬)']

    hover_text = df_sort['sku_name'] + ', ' + \
                 df_sort['鄉鎮市區'] + ', ' + \
                 df_sort['建案名稱'].astype(str) + ', ' + \
                 df_sort['交易年'].astype(str) + '年, ' + \
                 df_sort['sku_dist'].astype(int).astype(str) + '公尺, ' + \
                 df_sort['sku_109_total'].astype(int).astype(str) + '人'

    fig_sku_1 = make_subplots(rows=3, cols=2,
                              specs=[[{"rowspan": 2, "colspan": 2}, None], [{}, {}], [{}, {}]],
                              subplot_titles=(f'{SKUs}個鄰近小學 v.s.每坪單價(萬)',))
    fig_sku_1 = fn_gen_plotly_scatter(fig_sku_1, df_sort['sku_name'], y_data, row=1, col=1, margin=margin,
                                      color=color_set, text=hover_text)

    hover_txt1 = df_hl['sku_name'] + ', ' + \
                 df_hl['鄉鎮市區'] + ', ' + \
                 df_hl['建案名稱'].astype(str) + ', ' + \
                 df_hl['交易年'].astype(str) + '年, ' + \
                 df_hl['sku_dist'].astype(int).astype(str) + '公尺, ' + \
                 df_hl['sku_109_total'].astype(int).astype(str) + '人'

    fig_sku_1 = fn_gen_plotly_scatter(fig_sku_1, df_hl['sku_name'], df_hl['每坪單價(萬)'], row=1, col=1, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sku_2 = make_subplots(rows=2, cols=2,
                              specs=[[{}, {}], [{}, {}]],
                              subplot_titles=('鄰近小學距離(公尺) v.s.坪單價(萬)',
                                              '鄰近小學人數(人) v.s. 每坪單價(萬)'))

    fig_sku_2 = fn_gen_plotly_scatter(fig_sku_2, df_sort['sku_dist'], y_data, row=1, col=1, margin=margin,
                                      color=color_set, text=hover_text)

    fig_sku_2 = fn_gen_plotly_scatter(fig_sku_2, df_hl['sku_dist'], df_hl['每坪單價(萬)'], row=1, col=1, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sku_2 = fn_gen_plotly_scatter(fig_sku_2, df_sort['sku_109_total'], y_data, row=1, col=2,
                                      margin=margin, color=color_set, text=hover_text)

    fig_sku_2 = fn_gen_plotly_scatter(fig_sku_2, df_hl['sku_109_total'], df_hl['每坪單價(萬)'], row=1, col=2, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    return fig_sku_1, fig_sku_2


def fn_gen_analysis_building(df, target, color_by, margin=None, bc_name=None):
    if bc_name is None:
        bc_name = ['康寶日出印象']
    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin
    y_data = df[target]

    color_set, opacity = fn_set_color_by(color_by, df)

    hover_text = df['鄉鎮市區'] + ', ' + \
                 df['建案名稱'].astype(str) + ', ' + \
                 df['交易年'].astype(str) + '年, ' + \
                 df['建物坪數'].astype(int).astype(str) + '坪, ' + \
                 df['總樓層數'].astype(int).astype(str) + '樓'

    df_hl = df[df['建案名稱'].apply(lambda x: x in bc_name)]  # <--

    fig_sct_3 = make_subplots(rows=2, cols=2,
                              subplot_titles=(f'交易年 v.s. {target}', f'建物坪數 v.s. {target}',
                                              f'移轉層次 v.s {target}', f'總樓層數 v.s. {target}'))

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df['交易年'], y_data, row=1, col=1, margin=margin, color=color_set,
                                      text=hover_text, opacity=opacity)

    hover_txt1 = df_hl['鄉鎮市區'] + ',' + \
                 df_hl['建案名稱'].astype(str) + ',' + \
                 df_hl['交易年'].astype(str) + '年,' + \
                 df_hl['建物坪數'].astype(int).astype(str) + '坪,' + \
                 df_hl['總樓層數'].astype(int).astype(str) + '樓'

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df_hl['交易年'], df_hl[target], row=1, col=1, margin=margin, color='red',
                                      text=hover_txt1, opacity=1)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df['建物坪數'], y_data, row=1, col=2, margin=margin, color=color_set,
                                      text=hover_text, opacity=opacity)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df_hl['建物坪數'], df_hl[target], row=1, col=2, margin=margin,
                                      color='red',
                                      text=hover_txt1, opacity=1)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df['移轉層次'], y_data, row=2, col=1, margin=margin, color=color_set,
                                      text=hover_text, opacity=opacity)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df_hl['移轉層次'], df_hl[target], row=2, col=1, margin=margin,
                                      color='red',
                                      text=hover_txt1, opacity=1)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df['總樓層數'], y_data, row=2, col=2, margin=margin, color=color_set,
                                      text=hover_text, opacity=opacity)

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df_hl['總樓層數'], df_hl[target], row=2, col=2, margin=margin, color='red',
                                      text=hover_txt1, opacity=1)
    return fig_sct_3


def fn_gen_analysis_statistic(df):
    fig_bar = make_subplots(rows=2, cols=2, subplot_titles=('交易年', '交易月', '每坪單價(萬)', '總價(萬)'))
    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20}

    fig_bar = fn_gen_plotly_hist(fig_bar, df['交易年'], '交易年', row=1, col=1, bins=30, margin=margin)
    fig_bar = fn_gen_plotly_hist(fig_bar, df['交易月'], '交易月', row=1, col=2, bins=50, margin=margin)
    fig_bar = fn_gen_plotly_hist(fig_bar, df['每坪單價(萬)'], '單價(萬坪)', row=2, col=1, bins=50, margin=margin)
    fig_bar = fn_gen_plotly_hist(fig_bar, df['總價(萬)'], '總價(萬)', row=2, col=2, bins=50, margin=margin)

    fig_bar_2 = make_subplots(rows=2, cols=2, subplot_titles=('建物坪數', '總樓層數', '車位類別', '車位單價(萬)'))
    fig_bar_2 = fn_gen_plotly_hist(fig_bar_2, df['建物坪數'], '建物坪數', row=1, col=1, bins=50, margin=margin)
    fig_bar_2 = fn_gen_plotly_hist(fig_bar_2, df['總樓層數'], '總樓層數', row=1, col=2, bins=50, margin=margin)
    fig_bar_2 = fn_gen_plotly_hist(fig_bar_2, df['車位類別'], '車位類別', row=2, col=1, bins=50, margin=margin)
    fig_bar_2 = fn_gen_plotly_hist(fig_bar_2, df['車位單價(萬)'], '車位單價(萬)', row=2, col=2, bins=50, margin=margin)

    df_pk_1 = df[df['車位類別'] == '坡道平面']
    df_pk_2 = df[df['車位類別'] == '坡道機械']
    fig_bar_3 = make_subplots(rows=2, cols=2,
                              subplot_titles=('坡道平面 的 價格分佈', '坡道平面 的 坪數分佈', '坡道機械 的 價格分佈', '坡道機械 的 坪數分佈'))
    fig_bar_3 = fn_gen_plotly_hist(fig_bar_3, df_pk_1['車位單價(萬)'], '車位單價(萬)', row=1, col=1, bins=50, margin=margin)
    fig_bar_3 = fn_gen_plotly_hist(fig_bar_3, df_pk_1['車位坪數'], '車位坪數', row=1, col=2, bins=50, margin=margin)
    fig_bar_3 = fn_gen_plotly_hist(fig_bar_3, df_pk_2['車位單價(萬)'], '車位單價(萬)', row=2, col=1, bins=50, margin=margin)
    fig_bar_3 = fn_gen_plotly_hist(fig_bar_3, df_pk_2['車位坪數'], '車位坪數', row=2, col=2, bins=50, margin=margin)

    dists = len(df['鄉鎮市區'].unique())
    df_typ = df[df['都市土地使用分區'].apply(lambda x: x == '住' or x == '商')]
    fig_bar_4 = make_subplots(rows=2, cols=2, subplot_titles=(f'土地使用分區', f'行政區({dists}個)', '', ''))
    fig_bar_4 = fn_gen_plotly_hist(fig_bar_4, df_typ['都市土地使用分區'], '土地使用分區', row=1, col=1, bins=50, margin=margin)
    fig_bar_4 = fn_gen_plotly_hist(fig_bar_4, df['鄉鎮市區'], '行政區', row=1, col=2, bins=50, margin=margin)

    return fig_bar, fig_bar_2, fig_bar_3, fig_bar_4


def fn_gen_analysis(df, latest_records, build_case):
    config = {'scrollZoom': True,
              'toImageButtonOptions': {'height': None, 'width': None}}

    with st.expander(f'👓 檢視 每坪單價 的 分佈狀況'):
        fig_3d = px.scatter_3d(df, x='經度', y='緯度', z='每坪單價(萬)', color='每坪單價(萬)',
                               hover_data=['鄉鎮市區', '建案名稱', '交易年', 'MRT', 'sku_name'],
                               opacity=0.8)
        fig_3d.update_layout(title='每坪單價 的 分佈狀況', autosize=True,
                             width=700, height=500,
                             margin={'l': 0, 'r': 0, 't': 30, 'b': 20})
        st.plotly_chart(fig_3d)

        fig_c = go.Figure(data=go.Contour(x=df['經度'], y=df['緯度'], z=df['coor_ave'], line_smoothing=1.2))
        fig_c.update_layout(title='每坪單價的分佈狀況', autosize=True,
                            margin={'l': 50, 'r': 20, 't': 30, 'b': 20})
        st.plotly_chart(fig_c)

    with st.expander(f'👓 檢視 物件特徵的分布狀況'):
        fig_bar, fig_bar_2, fig_bar_3, fig_bar_4 = fn_gen_analysis_statistic(df)
        st.plotly_chart(fig_bar, config=config)
        st.plotly_chart(fig_bar_2, config=config)
        st.plotly_chart(fig_bar_3, config=config)
        st.plotly_chart(fig_bar_4, config=config)

    with st.expander(f'👓 檢視 每坪單價 與 "各項"指標 的關係'):
        # fig= plot_correlation(df,'每坪單價(萬))
        # st.write(fig)
        title = '每坪單價 與 "各項指標" 的關係'
        target = [dict(label='每坪單價', values=df['每坪單價(萬)'])]

        dimensions = [
            dict(label='通勤時間', values=df['MRT_Commute_Time_UL']),
            dict(label='捷運距離', values=df['MRT_DIST']),
            dict(label='進站人數', values=df['MRT_Tput_UL']),
            dict(label='出站人數', values=df['MRT_Tput_DL']),

            dict(label='小學距離', values=df['sku_dist']),
            dict(label='小學人數', values=df['sku_109_total']),
            dict(label='交易樓層', values=df['移轉層次']),
            dict(label='總樓層數', values=df['總樓層數']),

            dict(label='交易年度', values=df['交易年']),
            dict(label='建物坪數', values=df['建物坪數']),
            dict(label='經度', values=df['經度']),
            dict(label='緯度', values=df['緯度']),

            dict(label='座標平均', values=df['coor_ave']),
            dict(label='學區平均', values=df['SKU_ave']),
            dict(label='捷運平均', values=df['MRT_ave']),
            dict(label='行政區平均', values=df['DIST_ave']),
        ]

        figs = 4
        d1 = dimensions[:figs]
        d2 = dimensions[figs: 2 * figs]
        d3 = dimensions[2 * figs: 3 * figs]
        d4 = dimensions[3 * figs: 4 * figs]

        for d in [d1, d2, d3, d4]:
            fig = go.Figure(data=go.Splom(
                dimensions=d + target,
                diagonal=dict(visible=False),
                showupperhalf=False,
                marker=dict(color=df['每坪單價(萬)'],
                            size=6,
                            colorscale='Bluered',
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))))

            fig.update_layout(title=title,
                              dragmode='select',
                              width=800,
                              height=800,
                              hovermode='closest')

            st.plotly_chart(fig, config=config)

    with st.expander(f'👓 檢視 每坪單價 與 "捷運"指標 的關係'):
        color_by = st.radio('著色條件:', options=['無', '依捷運距離', '依通勤時間', f'依最新登錄({latest_records})'], index=0)
        fn_set_radio_2_hor()
        fig_sct, fig_sct_1 = fn_gen_analysis_mrt(df, color_by, bc_name=[build_case])
        st.plotly_chart(fig_sct, config=config)
        st.plotly_chart(fig_sct_1, config=config)

    with st.expander(f'👓 檢視 每坪單價 與 "小學"指標 的關係'):
        color_by = st.radio('著色條件:', options=['無', '依小學距離', '依小學人數', f'依最新登錄({latest_records})'], index=0)
        fn_set_radio_2_hor()
        fig_sku_1, fig_sku_2 = fn_gen_analysis_sku(df, color_by, bc_name=[build_case])
        st.plotly_chart(fig_sku_1, config=config)
        st.plotly_chart(fig_sku_2, config=config)

    with st.expander(f'👓 檢視 每坪單價 與 "建物"指標 的關係'):
        color_by = st.radio('著色條件:', options=['無', '依交易年', '依總樓層數', '依建物坪數', f'依最新登({latest_records})'], index=0)
        fn_set_radio_2_hor()
        fig_sct_3 = fn_gen_analysis_building(df, '每坪單價(萬)', color_by, bc_name=[build_case])
        st.plotly_chart(fig_sct_3, config=config)

    with st.expander(f'👓 檢視 物件總價 與 "建物"指標 的關係'):
        color_by = st.radio('著色條件:', options=['無', '依交易年', '依總樓層數', '依建物坪數', f'依最新登錄({latest_records})'], index=0,
                            key=1)
        fn_set_radio_2_hor()
        fig_sct_3 = fn_gen_analysis_building(df, '總價(萬)', color_by, bc_name=[build_case])
        st.plotly_chart(fig_sct_3, config=config)


def fn_gen_bc_deals(build_case, dic_df_show):
    if len(dic_df_show.keys()):
        deals = np.count_nonzero(dic_df_show['每坪單價(萬)'])
        st.write('')
        st.subheader(f'🏡 建案: {build_case}'
                     f'📝 登錄: {deals} 筆'
                     f'💰 總金額: {round((dic_df_show["總價(萬)"].values.sum()) / 10000, 2)} 億')

        r = st.radio('檢視選項:', options=['總價(萬)', '每坪單價(萬)', '建物坪數', '車位坪數', '總價-車位(萬)', '車位總價(萬)', '交易日期'], index=0)
        fn_set_radio_2_hor()

        df_show = dic_df_show[r] if r in dic_df_show.keys() else None
        assert df_show is not None, f'{r} not in dic_df_show {dic_df_show.keys()}'
        fmt = "{:.2f}" if r in ['每坪單價(萬)', '建物坪數', '車位坪數'] else None
        df_show = df_show.astype(int) if r == '交易日期' else df_show
        df_show_fig = df_show.style.format(fmt).applymap(fn_gen_df_color).highlight_max(axis=1, color='pink')
        df_show_fig = df_show_fig.background_gradient(cmap='rainbow', low=0.8, high=0, axis=None)
        st.dataframe(df_show_fig, width=768, height=540)
        dic_values = defaultdict(list)
        for col in df_show.columns:
            for idx in df_show.index:
                v = df_show.loc[idx, col]
                a = dic_df_show['建物坪數'].loc[idx, col]
                if v > 0:
                    dic_values[a].append(v)

        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=(
                                f'建案-{build_case}: {len(dic_values.keys())}種坪數 共{deals}筆交易 的 "{r}" 分布',))

        margin = {'l': 40}
        for k in dic_values.keys():
            fig = fn_gen_plotly_hist(fig, dic_values[k], f'{str(k)}坪{r}', bins=50, margin=margin,
                                     line_color='black', showlegend=True)
        st.plotly_chart(fig)


def fn_gen_model_confidence(loaded_model, X):
    preds = np.stack([t.predict(X.values) for t in loaded_model.estimators_])
    trees = preds.shape[0]
    stds = preds.std(0)
    preds_std = list(map(lambda x: round(x, 2), stds))
    conf = list(map(lambda x: round(100 - x, 2), preds_std))
    # preds_std = round(preds.std(0)[0],2)
    # conf = 100 - preds_std
    # print(X.shape, preds.shape, trees, stds.shape, len(preds_std), len(conf))

    return trees, conf


def fn_gen_web_eda(df):
    t_s = time.time()

    df_sel = df.copy()
    options = list(df_sel[['MRT']].sort_values(by='MRT')['MRT'].unique()) + ['不限']
    idx = options.index('R線_關渡站') if 'R線_關渡站' in options else 0
    mrt = st.sidebar.selectbox('捷運站', options=options, index=idx)
    df_sel = df_sel.reset_index(drop=True) if mrt == '不限' else df_sel[df_sel['MRT'] == mrt].reset_index(drop=True)

    build_cases = ['不限'] + [b for b in df_sel['建案名稱'].astype(str).unique()]
    build_cases.remove('nan') if 'nan' in build_cases else None
    build_case = st.sidebar.selectbox('建案名稱', options=build_cases, index=len(build_cases) - 1)
    df_sel = df_sel[df_sel['建案名稱'] == build_case].reset_index(drop=True) if build_case != '不限' else df_sel

    floor = st.sidebar.selectbox('移轉層次', (0, *df_sel['移轉層次'].unique()))
    df_sel = df_sel[df_sel['移轉層次'] == floor].reset_index(drop=True) if floor != 0 else df_sel

    From = str(df_sel['交易年月日'].iloc[-1])
    From = From[0:-4] + '年' + From[-4].replace('0', '') + From[-3] + '月'
    To = str(df_sel['交易年月日'].iloc[0])
    To = To[0:-4] + '年' + To[-4].replace('0', '') + To[-3] + '月'

    From_To = f'{From} 至 {To} 有{len(df_sel)}筆交易'
    ave = round(df_sel['每坪單價(萬)'].mean(), 0)

    # df_bc = pd.DataFrame()
    dic_df_show = dict()
    if build_case != '不限' and not build_case.endswith('區'):
        floor_max = df_sel['總樓層數'].max()
        floor_list = [str(floor_max - i) + 'F' for i in range(floor_max)]

        if len(df_sel['戶別'].unique()) == 1:
            df_sel['house_num'] = df_sel['土地位置建物門牌'].apply(fn_addr_2_house_num)
        else:
            df_sel['house_num'] = df_sel['戶別'].apply(lambda x: x.split('-')[0] if '-' in x else x)

        house_nums = sorted(df_sel['house_num'].unique())

        df_bc = pd.DataFrame(index=floor_list, columns=house_nums)
        df_bc_t = df_bc.copy()
        df_bc_car = df_bc.copy()
        df_bc_s = df_bc.copy()
        df_bc_ps = df_bc.copy()
        df_bc_d = df_bc.copy()

        df_sel_sort = df_sel.sort_values(by='交易年月日', ascending=True)
        # print(f'{df_sel[["移轉層次", "建物坪數"]]}')
        for idx in df_sel_sort.index:
            flr = str(df_sel_sort.loc[idx, '移轉層次']) + 'F'
            num = df_sel_sort.loc[idx, 'house_num']
            val, total, car, size, p_size, date = df_sel_sort.loc[
                idx, ['每坪單價(萬)', '總價(萬)', '車位總價(萬)', '建物坪數', '車位坪數', '交易年月日']]

            df_bc.at[flr, num] = round(val, 2)
            df_bc_t.at[flr, num] = total
            df_bc_car.at[flr, num] = car
            df_bc_s.at[flr, num] = size
            df_bc_ps.at[flr, num] = p_size
            df_bc_d.at[flr, num] = date
            # print(f'{flr}, {p_size}, {size}')

        df_bc.fillna(round(0, 1), inplace=True)
        df_bc_t.fillna(round(0, 1), inplace=True)
        df_bc_car.fillna(round(0, 1), inplace=True)
        df_bc_s.fillna(round(0, 1), inplace=True)
        df_bc_ps.fillna(round(0, 1), inplace=True)
        df_bc_d.fillna(round(0, 1), inplace=True)
        if floor != 0:
            df_bc = df_bc[df_bc.index == str(floor) + 'F']
            df_bc_t = df_bc_t[df_bc_t.index == str(floor) + 'F']
            df_bc_car = df_bc_car[df_bc_car.index == str(floor) + 'F']
            df_bc_s = df_bc_s[df_bc_s.index == str(floor) + 'F']
            df_bc_ps = df_bc_ps[df_bc_ps.index == str(floor) + 'F']
            df_bc_d = df_bc_d[df_bc_d.index == str(floor) + 'F']

        dic_df_show['每坪單價(萬)'] = df_bc[df_bc.sum(axis=1) > 0]
        dic_df_show['總價(萬)'] = df_bc_t[df_bc_t.sum(axis=1) > 0]
        dic_df_show['車位總價(萬)'] = df_bc_car[df_bc_car.sum(axis=1) > 0]
        dic_df_show['建物坪數'] = df_bc_s[df_bc_s.sum(axis=1) > 0]
        dic_df_show['車位坪數'] = df_bc_ps[df_bc_ps.sum(axis=1) > 0]
        # dic_df_show['建物-車位(坪)'] = dic_df_show['建物坪數'] - dic_df_show['車位坪數']
        dic_df_show['總價-車位(萬)'] = dic_df_show['總價(萬)'] - dic_df_show['車位總價(萬)']
        dic_df_show['交易日期'] = df_bc_d[df_bc_d.sum(axis=1) > 0] / 100
        # print(f'{dic_df_show["建物坪數"] }')

    floors = list(df_sel['移轉層次'].unique())
    floors.sort()
    prices = []
    deals = []
    for f in floors:
        price = int(df_sel[df_sel['移轉層次'] == f]['每坪單價(萬)'].mean())
        deal = len(df_sel[df_sel['移轉層次'] == f])
        prices.append(price)
        deals.append(deal)

    floors = [str(f) + 'F' for f in floors]
    fig_bar2 = go.Figure(data=[
        go.Bar(name='均價(萬/坪)', x=floors, y=prices),
        go.Bar(name='成交戶數', x=floors, y=deals)
    ],
        layout={'title': f'{mrt} ({From_To})'})

    fig_bar2.update_layout(barmode='group',  # One of 'group', 'overlay' or 'relative'
                           margin=dict(l=30, r=20, t=60, b=40),
                           paper_bgcolor="LightsteelBlue",
                           font=dict(size=16))

    df_sel.rename(columns={'log': 'lon'}, inplace=True)  # rename for st.map

    # df_sel['每坪單價(萬)']=df_sel['每坪單價(萬)'].astype(int)
    df_sel['MRT_DIST'] = df_sel['MRT_DIST'].astype(int)

    df_sel.rename(columns={'MRT': '捷運站', 'MRT_DIST': '捷運站距離(m)'}, inplace=True)

    cols = st.sidebar.multiselect('欄位選擇', df_sel.columns, default=['捷運站', '建案名稱', '移轉層次', '建物坪數', '車位坪數',
                                                                   '車位類別', '總價(萬)', '每坪單價(萬)', '車位單價(萬)',
                                                                   '交易年月日', '地址'])

    df_cols = df_sel[cols]

    width = []
    for col in df_cols.columns:
        w = df_cols[col].apply(lambda x: len(str(x)))
        w = max((*w, len(col)))
        width.append(w)

    fig_tbl = go.Figure(data=[
        go.Table(columnwidth=width,
                 header=dict(values=list(df_cols.columns),
                             align='center',
                             font_size=14),
                 cells=dict(values=[df_cols[[col]] for col in df_cols.columns],
                            align='center',
                            font_size=14)
                 )])
    fig_tbl.update_layout(margin=dict(t=2, b=2, l=1, r=1))

    house_typ = '預售屋' if len(df['建築完成年月'].unique()) == 1 else '中古屋'
    # city = df['土地位置建物門牌'].apply(lambda x:x.split('市')+'市')
    # city = city.unique()

    period = f"民國 {df['交易年'].min()}年 ~ {df['交易年'].max()}年"
    title = f'{period}: {df.shape[0]} 筆 {house_typ} 實價登錄資料'
    hover_name = '交易年月日'
    hover_data = ["MRT", "建案名稱"]
    color = '每坪單價(萬)'
    map_style = "carto-positron"  # "open-street-map"
    fig_map_all = fn_gen_plotly_map(df, title, hover_name, hover_data, map_style, color=color)

    latest_rel = '0211'
    records = int(df.shape[0] - np.count_nonzero(df['Latest']))
    latest_records = f'版本:{latest_rel} 有 {records}筆'
    city = list(df['city'].unique())
    cities = ''
    for c in city:
        cities = cities + c + ' '

    # rendering web view

    st.subheader(f'🏙️ {cities} {house_typ} 實價登錄分析')
    st.plotly_chart(fig_map_all)

    st.write('')
    st.subheader(f'📊 數據分析')
    fn_gen_analysis(df, latest_records, build_case)

    st.write('')
    st.subheader(f'🚇 捷運 {mrt.split("_")[-1]} 周邊')
    st.subheader(From_To)
    st.subheader(f'均價 {int(ave)} 萬/坪')
    st.write('資料來源: [内政部不動產交易實價查詢服務網(每月1、11、21 日發布)](https://plvr.land.moi.gov.tw/DownloadOpenData)')
    df_cols = df_cols.sort_values(by='移轉層次', ascending=False) if '移轉層次' in df_cols.columns else df_cols
    AgGrid(df_cols)

    fn_gen_bc_deals(build_case, dic_df_show)

    st.write('')
    st.subheader('🗺️ 建案位置')
    df_sel['每坪單價'] = df_sel['每坪單價(萬)'].apply(lambda x: str(x) + '萬/坪')

    title = '街景圖'
    hover_name = '建案名稱'
    hover_data = ['交易年', '總價(萬)', '每坪單價(萬)', '車位單價(萬)',
                  '車位類別', '移轉層次', '捷運站', '捷運站距離(m)', ]
    map_style = "open-street-map"
    fig_map = fn_gen_plotly_map(df_sel, title, hover_name, hover_data, map_style, zoom=14)

    st.plotly_chart(fig_map)
    st.subheader('📈 樓層均價 與 成交戶數')
    st.plotly_chart(fig_bar2)

    t_e = time.time()
    dur_t = round(t_e - t_s, 5)
    print(f'fn_gen_web_eda: {dur_t} 秒')


def fn_gen_web_ml_train(df, path):
    ts = time.time()

    ml_model = os.path.join(path, 'output/model')

    # df =df[df['建案名稱']!='康寶日出印象']

    if not os.path.exists(ml_model):
        os.makedirs(ml_model)
    model_file = os.path.join(ml_model, 'ml_model.sav')

    st.subheader('機器學習')
    st.markdown(
        f' {"#" * 0} 🌳 **隨機森林 迴歸器** ([RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))')
    st.markdown(
        f' {"#" * 0} 💪 **極限梯度提升 迴歸器** ([XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn))')
    st.write('')

    with st.form(key='Form1'):
        col1, col2, col3 = st.columns(3)
        col1.markdown('##### 資料篩選:')
        city = list(df['city'].unique())
        city_sel = col1.radio('城市篩選', tuple(city + ['不限']), index=city.index('台北市'))
        # typ_sel = col1.radio('建物型態tuple(['大樓(>=11F),華廈( <11F)',不限']),index=e)
        bypass_1F = col1.radio('排除特殊樓層 ?', ('排除1F交易', '包含1F交易'), index=0)
        drop_sel = col1.radio('排除極端目標 ?', ('排除極值(<1%)', '包含'), index=0)
        ano_det = col1.radio('排除異常資料?', ('無', '異常偵測並排除', '異常偵測'), index=0)

        col2.markdown('##### 模型選擇:')
        ml_model = col2.radio('模型選擇', ('RandomForestRegressor', 'XGBRegressor'), index=0)
        tune = col2.radio('調校方式', ('Manually', 'GridSearch 🐢', 'RandomizedSearch 🚧'), index=0)
        tune = tune.split(' ')[0]
        threads = col2.radio('執行緒數量', ('Single-Thread', 'Multi-Threads 💀'), index=0)
        threads = threads.split(' ')[0]
        n_jobs = 1 if threads == 'Single-Thread' else -1

        col3.markdown('##### 超參數調校:')

        split = col3.slider('測試樣本比例', min_value=0.1, max_value=0.5, step=0.05, value=0.2)

        if len(st.session_state['para']):
            dft_trees = st.session_state['para']['n_estimators']
            dft_depth = st.session_state['para']['max_depth']
        else:
            dft_trees = 700
            dft_depth = 100

        trees = col3.slider('要使用幾棵樹訓練(n_estimators)', min_value=1, max_value=1000, step=10, value=dft_trees)
        max_depth = col3.slider('每顆樹的最大深度(max_depth)', min_value=1, max_value=500, step=10, value=dft_depth)
        if ml_model == 'XGBRegressor':
            if 'eta' in st.session_state['para'].keys():
                dft_eta = st.session_state['para']['eta']
            else:
                dft_eta = 0.02

            eta = col3.slider('學習率 (eta)', min_value=0.01, max_value=0.3, step=0.01, value=dft_eta)

        mse_th = col3.slider('模型儲存門檻(MSE)', min_value=0., max_value=9., step=0.1, value=5.)

        st.write('')
        submitted = st.form_submit_button("上傳")

        if submitted:
            st.write(f'設定完成:')
            st.write(city_sel, bypass_1F, drop_sel, ano_det)
            st.write(ml_model, tune)
            if ml_model == 'XGBRegressor':
                st.write('樣本比:', split, '幾棵樹:', trees, '深度:', max_depth, '學習率:', eta)
            else:
                st.write('樣本比:', split, '幾棵樹:', trees, '深度:', max_depth)
        else:
            st.write('選擇參數後 請按 "上傳" 鍵')

    st.write('')
    dic_of_cat = {}

    # is_train = st.button('模型訓練')
    if True:  # if is_train or st.session_state['Train'] == 'done':
        # enc = OneHotEncoder()
        if bypass_1F == '排除1F交易':
            df = df[df['移轉層次'] > 1]  # bypass 1F since it's price is very special
            print(bypass_1F, df.shape[0])

        if city_sel != '不限':
            df = df[df['city'] == city_sel]
            print('選擇 ' + city_sel, df.shape[0])

        if drop_sel != '包含':  # Remove too few cat ( less than 1% )
            limit = float(drop_sel.split('%')[0][-1]) / 100
            # limit = 0.01
            df['cat'] = df['每坪單價(萬)'].apply(lambda x: int(x / 5))
            cat_count = dict(df['cat'].value_counts())
            total = df['cat'].shape[0]
            for k in cat_count.keys():
                if (cat_count[k] / total) < limit:
                    df = df[df['cat'] != k]
                    print(f'排除少於{limit * 100} % ({int(total * limit)}筆) 的目標: {k * 5}萬{cat_count[k]} 筆!')
            drop_num = total - df.shape[0]
            # st.write("')
            # st.markdown(f'{"#" * 5} 排除分少於{limit*100}%的目標:共{drop_num}筆')
            # st.write('')

        if '無' not in ano_det:
            df = fn_anomaly_detection(df.copy(), 100, 1)
            df_ano = df[df['ano']]

            if ano_det == '異常偵測並排除':
                df = df.drop(index=df_ano.index)

        grp = df['MRT'].value_counts()
        for idx in grp.index:
            if grp.loc[idx] < 2:
                print(idx, grp.loc[idx])
                df = df[df['MRT'] != idx]

        df.reset_index(drop=True, inplace=True)

        X, df_cat = fn_gen_training_data(df, path)
        y = df[['每坪單價(萬)']]

        with st.form(key='Form2'):
            st.markdown('##### 訓練特徵選擇:')
            features_sel = st.multiselect('特徵選擇:', X.columns,
                                          default=[c for c in X.columns if
                                                   '建材' not in c and '車位' not in c and c != 'MRT'])
            st.write('')
            form2_submitted = st.form_submit_button('選擇')

            if form2_submitted:
                st.write(f'選擇了{len(features_sel)}個特徵')
            else:
                st.write('選擇特徵後 請按 "選擇" 鍵')

        for c in X.columns:
            if X[c].isnull().values.any():
                print(c)
                print(X[c])
                assert False

    st.write('')
    is_train = st.button('模型訓練')
    if is_train or st.session_state['Train'] == 'done':
        X = X[features_sel]
        if drop_sel == '包含':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=X.loc[:, 'MRT'])
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=df['cat'])

        with st.expander(f'👓 檢視 資料篩選'):
            if drop_sel != '包含':
                st.markdown(f'{"#" * 5} 排除分佈少於{limit * 100} % 的目標: 共{drop_num}筆')

            if '無' not in ano_det:
                st.markdown(f'{"#" * 5} {ano_det} 的資料: 共{df_ano.shape[0]}筆')
                df_screen = df_ano[['MRT', '地址', '每坪單價(萬)', '交易年月日', '備註']]
                AgGrid(df_screen)

        with st.expander(f'👓 檢視 資料分佈'):
            watch = "每坪單價(萬)"
            st.markdown(f'{"#" * 5} 目標 *“{watch}"* 在 訓練 與 測試 樣本的分佈狀況:')
            fig = make_subplots(rows=2, cols=1)

            margin = dict(t=10, b=0, l=0, r=0)
            fn_gen_plotly_hist(fig, y_train[watch], '訓練', row=1, margin=margin)
            fn_gen_plotly_hist(fig, y_test[watch], '測試', row=2, margin=margin)
            st.plotly_chart(fig)

            if 'MRT' in X.columns:
                for col_2_check in ['MRT']:  # ,移轉層次,交易年
                    st.markdown(f'{"#" * 5} 特徵 *“{col_2_check}”* 在 訓練 與 測試 樣本的佈狀況:')
                    fig = make_subplots(rows=2, cols=1)
                    margin = dict(t=10, b=0, l=0, r=0)
                    fn_gen_plotly_hist(fig, X_train[col_2_check].sort_values(), f'訓練:{X_train.shape[0]} 筆', row=1,
                                       margin=margin)
                    fn_gen_plotly_hist(fig, X_test[col_2_check].sort_values(), f'測試: {X_test.shape[0]} 筆', row=2,
                                       margin=margin)
                    st.plotly_chart(fig)

        if 'MRT' in X.columns:
            X = X.drop(columns='MRT')
            X_train = X_train.drop(columns='MRT')
            X_test = X_test.drop(columns='MRT')

        if tune == 'GridSearch':

            if ml_model == 'XGBRegressor':
                regr_sel = xgb.XGBRegressor()

                param_grid = [
                    {'n_estimators': [400, 500],
                     'max_depth': [40],
                     'eta': [0.01, 0.02]},
                ]
            else:
                regr_sel = RandomForestRegressor()

                param_grid = [
                    {'n_estimators': [600, 800, 1000],
                     'max_depth': [50, 100, 150]},
                    # ('bootstrap': [False],
                    # 'n_estimators': [150,200,259].
                    # 'max_features': [10, X_train.shape[1]]),
                ]

            regr = GridSearchCV(regr_sel,
                                param_grid,
                                cv=10,
                                scoring='neg_mean_squared_error',
                                return_train_score=True,
                                refit=True,
                                n_jobs=n_jobs)
        else:
            if ml_model == 'XGBRegressor':
                regr = xgb.XGBRegressor(max_depth=max_depth,
                                        n_estimators=trees,
                                        objective='reg:squarederror',  # "reg: linear or squarederror"
                                        random_state=42,
                                        eta=eta)

            else:
                regr = RandomForestRegressor(max_depth=max_depth,
                                             n_estimators=trees,
                                             random_state=42,
                                             # criterion='squared_error',
                                             oob_score=True,
                                             max_features='auto')  # "auto", "sqrt", "log2"

        for c in X_train.columns:
            val = X_train[[c]].describe().loc['count', :].values[0]
            if X_train.shape[0] != val:
                print(f'should have {X_train.shape[0]} but feature {c} have {val} valid data only !!! ')
                print(X_train[[c]].describe())
                print(X_train[X_train[c].isna()][c])

        regr.fit(X_train, y_train.values.ravel())

        if tune == 'GridSearch':
            print(regr.best_params_)
            # st.write('')
            col2.text(f'{tune} Result:')
            col2.write(regr.best_params_)
            # st.write('')
            st.session_state['para'] = regr.best_params_

        st.session_state['Train'] = 'done'

        fn_gen_web_ml_eval(ml_model, model_file, regr, X_train, X_test, y_train, y_test, df, mse_th)

        # st.session_state['Train'] = 'done'

    te = time.time()
    dur = round(te - ts, 5)
    print(f'fn_gen_web_ml_train: {dur} 秒')


def fn_gen_web_ml_eval(ml_model, model_file, regr, X_train, X_test, y_train, y_test, df, mse_th):
    ts = time.time()
    # scores = cross_val_score(regr, x_train, y_train.values.ravel(),cv=5) # st.write(scores)
    pred_train = regr.predict(X_train)
    pred_test = regr.predict(X_test)

    dic_of_metric = {}
    dic_of_metric['樣本數'] = [len(y_train), len(y_test)]
    dic_of_metric['R2 score'] = [r2_score(y_train, pred_train), r2_score(y_test, pred_test)]
    dic_of_metric['MSE'] = [mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)]
    dic_of_metric['MAE'] = [mean_absolute_error(y_train, pred_train), mean_absolute_error(y_test, pred_test)]
    try:
        dic_of_metric['OOB score'] = [regr.oob_score_, np.nan] if ml_model == 'RandomForestRegressor' else [np.nan,
                                                                                                            np.nan]
    except:
        pass

    df_result = pd.DataFrame(dic_of_metric, index=['訓練集', '測試集']).T
    df_result['差異'] = df_result['測試集'] - df_result['訓練集']

    st.write('')
    is_model_save = st.button('訓練並儲存 模型')
    if is_model_save:
        df_F = pd.DataFrame()
        df_F['Features'] = X_train.columns
        # df_F.to_csv(model_file.replace('.sav', '.csv'), encoding='utf-8-sig', index=False)
        # pickle.dump(regr, open(model_file, 'wb'))
        mse = round(df_result.loc["MSE", "測試集"], 2)
        st.session_state['Model_Metrics'] = f'此 {ml_model} 模型在測試資料集MSE為 {mse}'
        # st.markdown(f'{"#" * 6} {st.session_state["Model_Metrics"]} 已儲存 💾 !')
        # st.write(f'save to {model_file}')
        date = datetime.datetime.today().date()
        # date = str(date.month)+str(date.day)
        date_str = str(date.month) if date.month > 9 else '0' + str(date.month)
        date_str += str(date.day) if date.day > 9 else '0' + str(date.day)
        # print(mse)
        if mse < mse_th:
            model_typ = 'xgb' if ml_model == 'XGBRegressor' else 'rf'
            city = 'all_city'
            if len(df['台北市'].unique()) == 1:
                city = 'tpe' if df['台北市'].unique() == 1 else 'new_tpe'

            good_model = model_file.split('.sav')[
                             0] + f'_{city}_{model_typ}_mse_{str(mse).replace(".", "p")}_{date_str}.sav'
            df_F.to_csv(good_model.replace('.sav', '.csv'), encoding='utf-8-sig', index=False)
            pickle.dump(regr, open(good_model, 'wb'))
            st.markdown(f'{"#" * 6} ✨ 🥇 ✨ 唉呦~ 不錯喔! 打上標籤 收藏起來: ml_{good_model.split("ml_")[-1]} 💾 !')
    else:
        # st.write( 訓練模型尚未儲存!
        st.markdown(f'{"#" * 6} 訓練型尚未儲存 !')

    st.write('')

    st.markdown(f'{"#" * 6} 訓練結果:')
    st.markdown(f'{"#" * 6} ⚆ 特徵數: {len(X_train.columns)} 個 (Features)')
    st.markdown(f'{"#" * 6} ⚆ 樣本數: {len(X_train) + len(X_test)} 筆 (Instances)')
    st.markdown(f'{"#" * 6} ⚆ 訓練指標(Metrics):')

    df_metrics = pd.DataFrame()
    df_metrics['實際價格'] = y_test['每坪單價(萬)']
    df_metrics['模型預估'] = pred_test
    df_metrics['模型預估'] = df_metrics['模型預估'].apply(lambda x: round(x, 2))

    df_metrics['誤差(萬/坪)'] = df_metrics['模型預估'] - df_metrics['實際價格']
    df_metrics['誤差(萬/坪)'] = df_metrics['誤差(萬/坪)'].apply(lambda x: round(x, 2))
    df_metrics['誤差(%)'] = round(100 * (df_metrics['模型預估'] - df_metrics['實際價格']) / df_metrics['實際價格'], 2)
    df_metrics = df_metrics.reset_index(drop=True)
    df_metrics['建案名稱'] = df[[idx in df_metrics.index for idx in df.index]]['建案名稱']
    df_metrics['地址'] = df[[idx in df_metrics.index for idx in df.index]]['地址']
    # df_metrics['樓層']=x[[idx in df_metrics.index for idx in x.index]][移轉層次]
    df_metrics['MRT'] = df[[idx in df_metrics.index for idx in df.index]]['MRT']

    del df

    c1, c2 = st.columns(2)
    c1.table(df_result)

    err_th = 2
    df_sel = df_metrics[df_metrics['誤差(萬/坪)'].apply(lambda x: abs(x) < err_th)]
    title = f'測試誤差分佈, 誤差<{err_th}萬的預測達{int(100 * df_sel.shape[0] / df_metrics.shape[0])}%'
    fig = make_subplots(rows=1, cols=1, subplot_titles=(title,))

    margin = dict(t=30, b=250, l=0, r=400)
    fig = fn_gen_plotly_hist(fig, df_metrics['誤差(萬/坪)'], '測試誤差分佈(萬)', margin=margin, opacity=0.7)
    fig = fn_gen_plotly_hist(fig, df_sel['誤差(萬/坪)'], '測試誤差分佈(萬)', margin=margin, bins=10, barmode='overlay',
                             opacity=0.7)

    c2.plotly_chart(fig)

    try:
        df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': regr.feature_importances_})
    except:
        df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': regr.best_estimator_.feature_importances_})

    df_imp = df_imp.sort_values(by='Importance')

    df_top = df_imp.iloc[df_imp.shape[0] - 10:df_imp.shape[0] + 1, :]
    df_bot = df_imp.iloc[:10, :]

    x_data_col = 'Importance'
    y_data_col = 'Features'
    color_col = 'Importance'
    text_col = 'Importance'
    v_or_h = 'h'
    margin = dict(t=0, b=0, l=10, r=15)
    text_fmt = '%{value:.5f}'

    fig_top = fn_gen_plotly_bar(df_top, x_data_col, y_data_col, text_col, v_or_h, margin,
                                color_col=color_col, text_fmt=text_fmt)

    fig_bot = fn_gen_plotly_bar(df_bot, x_data_col, y_data_col, text_col, v_or_h, margin,
                                color_col=color_col, text_fmt=text_fmt, ccs='haline')

    st.markdown(f'{"#" * 5} 根據訓練模型 影房價最 "重要" 的{df_top.shape[0]}個因素是?')
    st.plotly_chart(fig_top)

    st.markdown(f'{"#" * 5} 根據訓練模型 影房價較 "不重要" 的{df_bot.shape[0]}個因素是?')
    st.plotly_chart(fig_bot)

    st.write('測試資料集的模型預結果(萬/坪):')
    # st.dataframe(df_metrics)
    AgGrid(df_metrics)

    te = time.time()
    dur = round(te - ts, 5)
    print(f'fn_gen_web_ml_eva:{dur}秒')


def fn_gen_web_ml_inference(path, build_typ):
    ts = time.time()

    ml_model = os.path.join(path, r'output/model')
    if not os.path.exists(ml_model):
        os.makedirs(ml_model)

    # model_file = os.path.join(ml_model, 'ml_model.sav')

    model_fdr = ml_model
    models = []
    for i, j, files in os.walk(ml_model):
        for f in files:
            if '.sav' in f and f not in models:
                models.append(f)

    if len(models) > 0:
        st.write('')
        st.subheader('模型推論')

        model_sel = st.selectbox('模型選擇:', models)
        model_typ = model_sel.split('tpe')[-1].split('mse')[0].replace('_', '')
        model_sel = os.path.join(model_fdr, model_sel)

        # load the model from disk
        loaded_model = fn_load_model(model_sel)

        df_F = pd.read_csv(model_sel.replace('.sav', '.csv'), encoding='utf-8-sig')

        dic_of_input = {}
        with st.form('Form2'):
            # 台北市北投區西安街二段197號
            # 台北市北投區大度路三段301巷
            # 台北市北投區大度路三段301巷67號
            addr = st.text_input(label='物件地址', value='台北市北投區大度路三段301巷67號')

            addr = fn_addr_handle(addr)
            df_coor_read = fn_house_coor_read()

            # build case = fn_addr_2_build_case(addr)

            geo_info, is_coor_save, is_match, addr_fr_db = fn_get_geo_info(addr, df_coor_read, slp=5)

            st.write(f'鄰近地址: {addr_fr_db}') if is_match else None

            # mrt_info, addr_coor, sku_info
            if addr not in df_coor_read.index:
                dic_of_coor = {addr: (geo_info['coor']['lat'], geo_info['coor']['log'])}
                df_coor = pd.DataFrame(dic_of_coor, index=['lat', 'lon'])
                df_coor = df_coor.T

                dic_of_dist = fn_get_admin_dist(addr)
                for k in dic_of_dist.keys():
                    df_coor[k] = dic_of_dist[k]

                df_coor_save = df_coor_read.append(df_coor)
                if is_coor_save:
                    fn_house_coor_save(df_coor_save)

            dic_of_input['台北市'] = 1 if '台北市' in addr else 0

            for d in geo_info.keys():
                for k in geo_info[d].keys():
                    dic_of_input[k] = geo_info[d][k]

            ave_path = dic_of_path['database']
            df_sku_ave = pd.read_csv(os.path.join(ave_path, 'SKU_ave.csv'), index_col='sku_name')
            df_mrt_ave = pd.read_csv(os.path.join(ave_path, 'MRT_ave.csv'), index_col='MRT')
            df_dist_ave = pd.read_csv(os.path.join(ave_path, 'DIST_ave.csv'), index_col='鄉鎮市區')

            mrt = dic_of_input['MRT']
            dic_of_input['MRT_ave'] = df_mrt_ave.loc[mrt, '每坪單價(萬)']

            sku = dic_of_input['sku_name']
            sku = sku if sku in df_sku_ave.index else fn_get_neighbor(sku, df_sku_ave.index)
            dic_of_input['SKU_ave'] = df_sku_ave.loc[sku, '每坪單價(萬)']
            dist = addr.split('市')[-1].split('區')[0] + '區'
            dic_of_input['DIST_ave'] = df_dist_ave.loc[dist, '每坪單價(萬)']

            dic_of_input['緯度'] = dic_of_input.pop('lat')
            dic_of_input['經度'] = dic_of_input.pop('log')

            c1, c2, c3 = st.columns(3)
            dic_of_input['建物坪數'] = c1.text_input(label='建物坪數(不含車位):', value=24)
            dic_of_input['車位坪數'] = c2.text_input(label='車位坪數:', value=2.21)
            dic_of_input['土地坪數'] = c3.text_input(label='土地坪數', value='未使用')

            c1, c2, c3, c4 = st.columns(4)
            dic_of_input['交易年'] = c1.slider('交易年(民國)', min_value=100, max_value=130, step=1, value=110)
            dic_of_input['移轉層次'] = c2.slider('交易樓層', min_value=2, max_value=40, step=1, value=14)
            dic_of_input['總樓層數'] = c3.slider('建物總樓層', min_value=2, max_value=40, step=1, value=15)
            dic_of_input['屋齡'] = c4.slider('屋齢', min_value=0, max_value=20, step=1, value=0)

            c1, c2, c3, c4, c5 = st.columns(5)
            dic_of_input['幾房'] = c1.radio('幾房?', (1, 2, 3, 4, 5, 6), index=2)
            dic_of_input['幾廳'] = c2.radio('幾廳?', (1, 2, 3, 4, 5, 6), index=0)
            dic_of_input['幾衛'] = c3.radio('幾衛?', (1, 2, 3, 4, 5, 6), index=1)
            dic_of_input['主要建材'] = c4.radio('建築結構', fn_get_categories(path, '主要建材'))
            dic_of_input['車位類別'] = c5.radio('車位類別', fn_get_categories(path, '車位類別'),
                                            index=fn_get_categories(path, '車位類別').index('坡道機械'))

            submitted = st.form_submit_button("上傳")

            if submitted:
                st.write(f'設定完成')
            else:
                st.write("選擇參數後按上傳鍵")

            st.session_state['pred_para'] = dic_of_input
            # st.write(st.session_state['pred_para'l)

        df_input = pd.DataFrame(dic_of_input, index=[0])
        X, df_cat = fn_gen_training_data(df_input, path, is_inference=True, df_F=df_F)
        # st.write(X)

        is_pred = st.button('click to predict')
        if is_pred:
            pred = loaded_model.predict(X)
            if model_typ == 'rf':
                trees, conf = fn_gen_model_confidence(loaded_model, X)

            price = round(pred[0], 2)
            area = float(dic_of_input["建物坪數"])
            total = int(price * area)

            with st.expander(f'Feature Number : {X.shape[1]}'):
                st.write(dict(X.iloc[0, :]))

            df_map = X.copy()
            df_map['預估單價'] = [f'預估單價: {str(price)} 萬坪']
            df_map['鄰近捷運站'] = [df_input['MRT'].values]
            df_map['捷運通勤時間'] = [str(X['MRT_Commute_Time_UL'].values[0]) + ' 分']
            df_map['捷運站距離'] = [str(X['MRT_DIST'].values[0]) + ' 公尺']
            df_map['鄰近小學'] = [df_input['sku_name'].values]
            df_map['小學距離'] = [str(X['sku_dist'].values[0]) + ' 公尺']
            df_map['小學人數'] = [str(int(X['sku_total'].values[0])) + ' 人']

            title = addr
            hover_data = ['鄰近捷運站', '捷運站距離', '捷運通勤時間', '鄰近小學', '小學距離', '小學人數']

            build_case = fn_addr_2_build_case(addr)
            if build_case != 'No_build_case_found':
                df_map['建案名稱'] = [build_case]
                title += f' ({build_case})'
                hover_data = ['建案名稱'] + hover_data

            # "open-street-map", "white-bg", "carto-positron", "stamen-terrain"
            fig = fn_gen_plotly_map(df_map, title, '預估單價', hover_data, 'carto-positron', zoom=15)

            st.subheader(f'模型預估:')
            st.markdown(f'{"#" * 4} 🔮 預估單價: {str(price)} 萬/坪')
            st.markdown(f'{"#" * 4} 🔮 預估總價: {total} 萬+車位價格')
            if model_typ == 'rf':
                show = '👍' if conf[0] > 96 else '👎'
                st.markdown(f'{"#" * 4} 🔮 信心指標: {conf[0]} {show}')

            st.plotly_chart(fig)

            if model_typ == 'rf':
                fn_gen_model_explain(X.copy(), loaded_model)

        is_rf = model_typ == 'rf'
        fn_gen_pred(path, loaded_model, model_sel, df_F, build_typ, is_rf)
    else:
        st.write(f'No models found in {model_fdr}')
        st.write('請先進行"模型訓練')

    te = time.time()
    dur = round(te - ts, 5)
    print(f'fn_gen_web_inference: {dur} 秒')


def fn_gen_web_init(path, page=None):
    print('fn_gen_web_init start')
    path_output = os.path.join(path, r'output')
    path_output = os.path.join(path_output, r'house_all.csv')
    print(path_output)
    if not os.path.exists(path_output):
        assert path_output + ' NOT existed !!!'
    # Initialization
    if 'Train' not in st.session_state:
        st.session_state['Train'] = 'not_yet'

    if 'Model_Metrics' not in st.session_state:
        st.session_state['Model_Metrics'] = 'init'

    if 'para' not in st.session_state:
        st.session_state['para'] = {}

    if 'pred_para' not in st.session_state:
        st.session_state['pred_para'] = {}

    # print(f'session_state: {st.session_state}')
    df = fn_get_house_data(path_output)
    df = fn_cln_house_data(df.copy())

    if page == 'train':
        cat_features = ['鄉鎮市區', '主要建材', '車位類別', 'MRT']
        for cat in cat_features:
            df_cat = pd.DataFrame(columns=[cat], data=sorted(list(df[cat].unique())))
            file = os.path.join(path, f'output\\Feature_{cat}.csv')
            df_cat.to_csv(file, encoding='utf-8-sig')

    print('fn_gen_web_init done')
    return df.copy()


def fn_gen_web_ref():
    st.subheader('數據來源:')
    st.write("- 實價登錄: [內政部 - 不動產成交案件 資料供應系統(每月1、11、21日發布)](https://plvr.land.moi.gov.tw/DownloadOpenData)")
    st.write("- 運輸資料流通服務平台: [交通部 - TDX(Transport Data eXchange)](https://tdx.transportdata.tw/)")
    st.write("- 座標資訊: [台灣電子地圖服務網](https://www.map.com.tw/)")
    st.write("- 座標資訊: [TGOS全國門牌地址定位服務](https://www.tgos.tw/tgos/Web/AddrssTGOS_Address.aspx)")
    st.write("- 國土資訊圖資服務平臺: [Taiwan Geospatial One Stop. 稱TGOS平臺](https://www.tgos.tw/tgos/web/tgos_home.aspx)")
    st.write("- 捷運 - 各站地址: [台北捷運 - 路網圖 各站資訊及時刻表](https://www.metro.taipei/cp.aspx?n=91974F2B13D997F1)")
    st.write("- 捷運 - 行駛時間: [台北捷運 - 單一車站至所有車站時間](https://web.metro.taipei/pages/tw/ticketroutetimesingle/068)")
    st.write("- 捷運 - 人數統計: [政府資料開放平台 - 台北捷運每日分時各站OD流量統計資料](https://data.gov.tw/dataset/128506)")
    st.write(
        "- 小學 - 人數統計: [統計處 - 各級學校基本資料](https://depart.moe.edu.tw/ed4500/News.aspx?n=5A930C32CC6C3818&sms=91B3AAE8C6388B96)")
    st.write("- 小學 - 各校地址(台北市): [政府資料開放平台 - 台北市各級學校分布圖](https://data.gov.tw/dataset/121225)")
    st.write("- 小學 - 各校地址(新北市): [政府資料開放平台 - 新北市學校通訊資料](https://data.gov.tw/dataset/123020)")
    st.write("- 歴史利率: [臺灣銀行存放款利率歷史資料表](https://www.cbc.gov.tw/tw/public/data/a13rate.xls)")
    st.write("- 歷史匯率: [臺灣期貨交易所 - 每日外幣参考匯率查詢](https://www.taifex.com.tw/cht/3/dailyFXRate)")
    st.write("- 經濟成長: [行政院主計總處 - 中華民國統計資訊網]")
    st.write("- 人口普查: [行政院主計總處 - 109年普查初步統計結果表](https://www.stat.gov.tw/ct.asp?mp=4&xItem=47698&ctNode=549)")
    st.write("- 單點坐標回傳行政區: [政府資料開放平台 - 坐標回傳行政區API](https://data.gov.tw/dataset/101898)")
    st.write("- 鄉鎮市區界線: [政府資料開放平台 - 我國各鄉(鎮·市、區)行政區城界線圖資](https://data.gov.tw/dataset/441)")
    st.write("- 村里界圖: [政府資料開放平台 - 各縣市村(里)界](https://data.gov.tw/dataset/7438)")

    st.write("")
    st.subheader('參考網站:')
    st.write("- 實價登錄網站: [樂居](https://www.leju.com.tw/)")
    st.write("- 實價登錄網站: [實價登錄比價王](https://community.houseprice.tw/building/118031)")
    st.write("- 房價預測網站: [中信銀行 智慧價平台](https://www.ctbcbank.com/contet/dam/minisite/long/loan/ctbc-mortgage/index.html)")
    st.write("- 房價預測網站: [好時價House+(利用統計學、數學及人工智慧(AI)演算法,算出不動產價值)](https://www.houseplus.tw/)")
    st.write("- 房價指數: [國立清華大學 安富金融工程研究中心](https://aife.site.nthu.edu.tw/p/404-1389-220340.php)")

    st.write('')
    st.subheader('相關競賽:')
    st.write("- 交通部: [交通數據創新應用競賽](https://tdx-contest.tca.org.tw) [TDX交通資料育成網](https://startup.transportdata.tw/)")
    st.write("- 玉山人工智慧公開挑戰賽2019夏季賽:[台灣不動產AI神預測](https://tbrain.trendmicro.com.tw/competitions/Details/6)")
    st.write("- 經濟部中小企業處:[2021城市數據實境賽](https://data.startupterrace.tw/data-contest)")

    st.write("")
    st.subheader('專利:')
    st.write("- 智能不動產估價專利: [中華民國專利資訊檢索系統](https://twpat2.tipo.gov.tw/twpatc/twpatkm?@@642176895)")


def fn_gen_web_tools():
    st.write("")
    st.subheader('機器學習:')
    st.write(
        "- 教科書: [Hands on Machine Learning - 第二章: 美國加州房價預測](https://nbviewer.org/github/DeqianBai/Hands-on-Machine-Learning/blob/master/02_Housing.ipynb)")
    st.write(
        "- 碩士論文: [淡江大學碩士在職專班 應用人工智慧於房價預測模型研究與分析(2019)](https://etds.lib.tku.edu.tw/ETDS/Home/Detail/U0002-2608201910580000)")
    st.write("- 模型應用: [微軟時空預測模型 FOST(Forecasting open source tool)]")

    st.write('')
    st.subheader('網頁製作:')
    st.write("- 純Python的極速網頁製作套件: [Streamlit](https://streamlit.io/)")
    st.write(
        "- Streamlit multi page framework: [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps)")
    st.write("- 畫文字 表情符號: [Emojipedia](https://emojipedia.org/)")

    st.write('')
    st.subheader('函式庫:')
    st.write("- 自動化資料分析: [DataPrep](https://docs.dataprep.ai/user_guide/user_guide.html)")
    st.write("- 網頁爬蟲自動化: [Selenium](https://www.selenium.dev/documentation/webdriver/)")
    st.write("- 好看的網頁表格: [Streamlit-Aggrid](https://github.com/PablocFonseca/streamlit-aggrid)")
    st.write("- 互動式網頁圖表: [Plotly](https://plotly.com/python/)")
    st.write("- 座標的距離計算: [GeoPy](https://geopy.readthedocs.io/en/stable/)")
    st.write("- 地理空間函式庫: [GeoPandas](https://geopandas.org/en/stable/)")
    st.write("- 中文轉阿拉伯數字: [cn2an](https://github.com/Ailln/cn2an)")
    st.write("- 瀑布圖: [waterfall_chart](https://github.com/chrispaulca/waterfall)")
    st.write("- 行事曆: [workalendar](https://github.com/workalendar/workalendar)")

    st.write('')
    st.subheader('其它工具:')
    st.write("- 圖轉文字: [LINE OCR](https://www.tech-girlz.com/2021/01/line-ocr.html)")
    st.write("- 圖轉CSV: [誠華OCR](https://zhtw.109876543210.com/)")


def fn_chrome_96_workaround():
    # st.write('<style>div{font-weight: normal;}</style>', unsafe_allow_html=True)
    pass


def fn_app(page='data'):
    print(f'fn_app() start, page = {page}')
    fn_chrome_96_workaround()
    # st.legacy_caching.clear_cache()

    st.sidebar.header(f'🔍 資訊篩選:\n')
    year_sel = st.sidebar.slider('交易年(民國)', min_value=100, max_value=111, value=(100, 111))
    c1, c2 = st.sidebar.columns(2)
    sel = c1.selectbox('交易類別', ['預售屋', '中古屋'], index=0)
    root = dic_of_path['root']
    path = os.path.join(root, r'pre_sold_house') if sel == '預售屋' else os.path.join(root, r'pre_owned_house')
    ml_model = os.path.join(path, r'output\model')

    if not os.path.exists(ml_model):
        os.makedirs(ml_model)

    # page = 'eda'
    if page == 'eda':
        df = fn_gen_web_init(path)
        df = df[df['交易年'].apply(lambda x: year_sel[0] <= x <= year_sel[1])]
        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)
        df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df

        c1, c2 = st.sidebar.columns(2)
        city = c1.selectbox('城市', ['台北市', '新北市', '不限'], index=0)
        is_tpe = city == '台北市'
        df = df[df['台北市'] == is_tpe] if city != '不限' else df

        d = c2.selectbox('鄉鎮市區', ['不限'] + df['鄉鎮市區'].unique().tolist(), index=0)
        df = df[df['鄉鎮市區'] == d] if d != '不限' else df

        land_typ = st.sidebar.selectbox('土地分區', ['不限', '住', '商'], index=0)
        df = df[df['都市土地使用分區'] == land_typ] if land_typ != '不限' else df

        fn_gen_web_eda(df)

    elif page == 'train':
        df = fn_gen_web_init(path, page=page)
        df = df[df['交易年'].apply(lambda x: year_sel[0] <= x <= year_sel[1])]
        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)
        df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df

        land_typ = st.sidebar.selectbox('土地分區', ['不限', '住', '商'], index=0)
        df = df[df['都市土地使用分區'] == land_typ] if land_typ != '不限' else df

        fn_gen_web_ml_train(df, path)

    elif page == 'inference':
        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)
        fn_gen_web_ml_inference(path, build_typ)

    elif page == 'reference':
        fn_gen_web_ref()

    elif page == 'tools':
        fn_gen_web_tools()

    else:
        st.write(f' page: {page} unhandle yet !!!')

    print(f'fn_app() done, page = {page}')
