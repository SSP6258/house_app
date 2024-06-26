import random
import time
import pandas as pd
import numpy as np
import os
import datetime
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors
import xgboost as xgb
import pickle
import jinja2
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
# from dataprep.eda import plot_correlation
from house_utils import fn_get_geo_info, fn_get_admin_dist, dic_of_path, geodesic, fn_get_coor_fr_db, fn_profiler, \
    fn_read_shp
from house_elt import fn_addr_handle, fn_house_coor_read, fn_house_coor_save
from house_elt import fn_gen_build_case, fn_gen_house_data
import plost
from streamlit.components.v1 import html
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
from ipyvizzustory import Story, Slide, Step

try:
    from streamlit_player import st_player
except:
    pass

# pip list --format=freeze > requirements.txt
dic_of_dbg = {'is_dbg': False,
              'time': 0,
              'sum': 0,
              'is_bypass_date_chk': True}

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
                  '使用分區_住': 'land_typ',
                  '稅_中位數': 'tax_median',
                  '稅_平均數': 'tax_mean',
                  '地下樓層': 'underground floor',
                  '總戶數': 'total households',
                  '基地面積(坪)': 'base area',
                  }


def fn_show_img(IMG_path, IMG_file, is_sidebar=False, width=None, caption=None):
    png = os.path.join(IMG_path, IMG_file)
    img = png if IMG_file.endswith('.gif') or IMG_file.startswith('http') else Image.open(png)
    # img = Image.open(png) if IMG_file.endswith('.png') or IMG_file.endswith('.jpeg') else png
    st.write('')
    if is_sidebar:
        st.sidebar.image(img, width=width, caption=caption)
    else:
        st.image(img, width=width, caption=caption)


def fn_addr_2_house_num(x):
    num = x.split('路')[-1]
    num = num.split('街')[-1]
    num = num.split('段')[-1]
    num = num.split('巷')[-1]
    num = num.split('弄')[-1]

    if num == x:
        print(x, 'special addr !!!')

    return num


@fn_profiler
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
@fn_profiler
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


def fn_house_hold_rename(x):
    y = ''
    for i in str(x):
        i = str(int(i)) if i.isnumeric() else i
        y += i

    y = y.replace(' ', '')

    # if x != y:
    #     print(f'{x} --> {y}')

    return y


@fn_profiler
def fn_cln_house_data(df):
    df['city'] = df['土地位置建物門牌'].apply(lambda x: x.split('市')[0].replace('臺', '台') + '市')
    df['建物移轉坪數'] = df['建物移轉坪數'].apply(lambda x: round(x, 2))
    df['建物型態'] = df['建物型態'].apply(lambda x: '華廈' if '華廈' in x else '大樓' if '大樓' in x else x)
    df.rename(columns={col: col.replace('移轉坪數', '坪數') for col in df.columns}, inplace=True)

    df = df[df['車位總價元'].astype(float) > 0] if '車位總價元' in df.columns else df
    df = df[df['里'].apply(lambda x: str(x).endswith('里'))] if '里' in df.columns else df
    df = df[df['稅_中位數'].apply(lambda x: str(x) != 'nan')] if '稅_中位數' in df.columns else df
    df['戶別'] = df['戶別'].apply(fn_house_hold_rename)

    df = fn_gen_build_case(df)
    df[['經度', '緯度']] = df[['log', 'lat']]

    return df


@st.cache(allow_output_mutation=True)
def fn_load_model(model_sel):
    try:
        loaded_model = pickle.load(open(model_sel, 'rb'))
    except:
        st.write(f'fn_load_model Fail: {model_sel}')
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


# @st.cache
@fn_profiler
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

    df.drop_duplicates(subset=['地址', '交易年月日', '總樓層數', '移轉層次', '每坪單價(萬)', '建物移轉坪數', '總價(萬)', '車位總價(萬)', '戶別'],
                       keep="first",
                       inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'Read {read_typ} data from {path} !!!')
    return df


@fn_profiler
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


@fn_profiler
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
                total = 'nan'
                year_total = f'{year - y}_Total'

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

    last_month = datetime.date.today().month - 1
    last_month = 12 if last_month == 0 else last_month
    sel_yr = df['交易年'].values[0] - 1 if last_month == 12 else df['交易年'].values[0]

    # print(f'last_month {last_month} sel_yr {sel_yr}')

    sel_month = last_month - 1 if last_month > 1 else last_month  # ToDo
    df['交易年月日'] = sel_yr * 10000 + int(sel_month) * 100 if '交易年月日' not in df.columns else df['交易年月日']

    df_rate = pd.read_csv(file, encoding='utf-8-sig', header=4)
    rate_col = df_rate.columns[13]  # 定存利率
    date_col = df_rate.columns[0]
    for idx in df.index:
        trade_date = float(int(df.loc[idx, '交易年月日'] / 100))

        if dic_of_dbg['is_bypass_date_chk']: # @ 240517
            if trade_date in df_rate[date_col].values:
                df_t = df_rate[df_rate[date_col] <= trade_date]
            else:
                df_t = df_rate.copy()

            rates = df_t[rate_col].values
            rate_sel = []
            for m in range(months):
                try:
                    rate_sel.append(rates[-1 - m])
                except:
                    rate_sel.append(rate_sel[-1])

                df.at[idx, f'利率_{m}個月前'] = rate_sel[-1]
        else:

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
                assert False, f'can NOT find interest_rate of {trade_date} {df.loc[idx, "交易年月日"]} {last_month} {sel_yr}'

    return df


def fn_get_categories(path, feature):
    df_f = pd.read_csv(os.path.join(path, f'output\\Feature_{feature}.csv'))
    cats = df_f[feature].to_list()

    return cats


def fn_get_hover_text(df):
    txt = ''
    cols = df.columns

    if '交易年' in cols:
        txt += df['交易年'].astype(str) + '年<br>'

    if '鄉鎮市區' in cols:
        txt += df['鄉鎮市區'].astype(str) + ' '

    if '里' in cols:
        txt += df['里'].astype(str) + '<br>'

    if '稅_平均數' in cols:
        txt += '所得平均 ' + (df['稅_平均數'] / 10).astype(int).astype(str) + ' 萬元<br>'

    if '稅_中位數' in cols:
        txt += '所得中位 ' + (df['稅_中位數'] / 10).astype(int).astype(str) + ' 萬元<br>'

    if '稅_平均數(萬)' in cols:
        txt += '所得平均 ' + (df['稅_平均數(萬)']).astype(int).astype(str) + ' 萬元<br>'

    if '稅_中位數(萬)' in cols:
        txt += '所得中位 ' + (df['稅_中位數(萬)']).astype(int).astype(str) + ' 萬元<br>'

    if '建案名稱' in cols:
        bc = df['建案名稱'].astype(str)
        bc = bc.apply(lambda x: '' if 'nan' in x else x)
        txt += bc + '<br>'

    if '移轉層次' in cols:
        txt += df['移轉層次'].astype(int).astype(str) + ' / '

    if '總樓層數' in cols:
        txt += df['總樓層數'].astype(int).astype(str) + 'F<br>'

    if '建物坪數' in cols:
        txt += df['建物坪數'].astype(int).astype(str) + '坪<br>'

    if 'MRT' in cols:
        txt += df['MRT'].astype(str) + ', '

    if 'MRT_DIST' in cols:
        txt += df['MRT_DIST'].astype(int).astype(str) + '公尺<br>'

    if 'MRT_Commute_Time_UL' in cols:
        txt += '通勤' + df['MRT_Commute_Time_UL'].astype(str) + '分<br>'

    if 'sku_name' in cols:
        txt += df['sku_name'].astype(str) + ', '

    if 'sku_dist' in cols:
        txt += df['sku_dist'].astype(int).astype(str) + '公尺<br>'

    if '投資建設' in cols:
        txt += df['投資建設'].astype(str) + '<br>'

    if '營造公司' in cols:
        txt += df['營造公司'].astype(str) + '<br>'

    if '每坪單價(萬)' in cols:
        txt += '每坪單價 ' + df['每坪單價(萬)'].astype(str) + ' 萬元<br>'

    return txt


@fn_profiler
def fn_gen_pred(path, model, model_name, df_F, build_typ, is_rf):
    st.write('')
    # st.subheader('批次驗證')
    st.markdown('#### :blue[$批次驗證:$]')
    st.markdown("$驗證資料:$ [$内政部不動產成交案件 資料供應系統(每月1、11、21日發布)$](https://plvr.land.moi.gov.tw/DownloadOpenData)")

    df_tax = pd.read_csv(os.path.join(dic_of_path['database'], '108_165-A.csv'), index_col='行政區')
    df_bc = pd.read_csv(os.path.join(dic_of_path['database'], 'build_case_info_ext.csv'), index_col='建案名稱')

    st.markdown('$資料上傳:$')
    file = st.file_uploader("資料上傳", type=['csv'], label_visibility='collapsed')
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
        df = fn_gen_house_data(os.path.join(temp, file.name), 'test', df_validate=df, is_trc=False)

        for m in df['MRT'].unique():
            if m in df_mrt_ave.index:
                pass
            else:
                st.error(f'{m} not in MRT_ave ! Remove {df[df["MRT"] == m].shape[0]} raws !')
                df = df[df['MRT'] != m]

        try:
            df['MRT_ave'] = df['MRT'].apply(lambda x: df_mrt_ave.loc[x, '每坪單價(萬)'])
        except:
            st.write(df['MRT'])
            st.write(df_mrt_ave)

        try:
            df['SKU_ave'] = df['sku_name'].apply(lambda x: df_sku_ave.loc[x, '每坪單價(萬)'])
        except:
            for s in df['sku_name'].values:
                if s not in df_sku_ave.index:
                    st.error(f'School {s} not in sku_ave.csv ?')

            assert False, f'Some School NOT in SKU_ave.csv'

        df['DIST_ave'] = df['鄉鎮市區'].apply(lambda x: df_dist_ave.loc[x, '每坪單價(萬)'])

        df_coor_read = fn_house_coor_read()
        for idx in df.index:
            addr = df.loc[idx, '地址']
            addr = addr if '台北市' in addr else '台北市' + addr

            vill = ''
            if addr in df_coor_read.index:
                vill = df_coor_read.loc[addr, '里']

            if vill == '' or str(vill) == 'nan':
                try:
                    addr_coor, is_match, add_fr_db = fn_get_coor_fr_db(addr, df_coor_read.copy(), is_trc=False)
                except:
                    is_match = False

                if is_match:
                    vill = df_coor_read.loc[add_fr_db, '里']
                    # print(addr, 'not in coor addr, try', add_fr_db, vill)

            if str(vill).endswith('里'):
                df.at[idx, '里'] = vill

        if '里' in df.columns:
            df = df[df['里'].astype(str) != 'nan']
            df = df[df['里'].apply(lambda x: x in df_tax['里'].values)]
            df['稅_中位數'] = df['里'].apply(lambda x: df_tax[df_tax['里'] == x]['中位數'].values[0])
            df['稅_平均數'] = df['里'].apply(lambda x: df_tax[df_tax['里'] == x]['平均數'].values[0])
        else:
            assert False, f'No 里 in df.columns'

        if '建案名稱' in df.columns:
            for idx in df.index:
                bc = df.loc[idx, '建案名稱']
                if bc in df_bc.index:
                    for c in ['地下樓層', '總戶數', '基地面積(坪)']:
                        df.at[idx, c] = df_bc.loc[bc, c]
        else:
            assert False, f'No 建案名稱 in df.columns'

        for c in ['地下樓層', '總戶數', '基地面積(坪)']:
            df = df[df[c].apply(lambda x: str(x).split('.')[0].isnumeric())]
            assert df.shape[0] > 0, f'{c} df.shape = {df.shape}'
            df[c] = df[c].apply(lambda x: int(round(float(x), 0)))

        if df.shape[0] and n_data:
            df = fn_cln_house_data(df.copy())

            df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df
            for i in df['主要建材'].tolist():
                if i not in df['主要建材'].unique():
                    print(i)

            X, df_cat = fn_gen_training_data(df.copy(), path, is_inference=True, df_F=df_F)
            try:
                df['模型預估(萬/坪)'] = model.predict(X)
            except:
                print(X.dtypes)
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

            if True:  # is_rf:
                config = {'scrollZoom': True,
                          'toImageButtonOptions': {'height': None, 'width': None}}

                st.write('')
                st.subheader(f'模型可信度分析')
                c1, c2 = st.columns(2)
                if is_rf:
                    ths = c2.slider('信心門檻', min_value=90, max_value=100, value=(96, 100))
                    th_l, th_h = ths[0], ths[1]
                    df_sel = df[df['信心指標'].apply(lambda x: th_h >= x >= th_l)]
                else:
                    # df['信心指標'] = df['鄉鎮市區'].apply(lambda x: 100)
                    df_sel = df

                colors = ['無', f'依行政區({len(df["鄉鎮市區"].unique())})', '依捷運距離', '依通勤時間']
                color_by = c1.selectbox('著色條件:', colors)

                margin = dict(t=50, l=50, r=0, b=50)
                fig = make_subplots()
                color_set, opacity = fn_set_color_by(color_by, df)
                hover_text = fn_get_hover_text(df)

                if is_rf:
                    title = f'模型: ml_model{model_name.split("ml_model")[-1]} 的 可信度評估 <br>' \
                            f'( 此模型進行{df.shape[0]}筆預測, 信心指標介於 {th_l} ~ {th_h} ' \
                            f'的 有{df_sel.shape[0]}筆, 約{int(100 * df_sel.shape[0] / df.shape[0])}% )'
                else:
                    title = f'模型: ml_model{model_name.split("ml_model")[-1]} 的 可信度評估'

                fig = fn_gen_plotly_scatter(fig, df['每坪單價(萬)'], df['模型預估(萬/坪)'], margin=margin,
                                            color=color_set, text=hover_text, opacity=0.6,
                                            xlabel='實際單價(萬/坪)', ylabel='預估單價(萬/坪)', title=title)
                color_set, opacity = fn_set_color_by(color_by, df_sel)
                hover_text = fn_get_hover_text(df_sel)

                if is_rf:
                    fig = fn_gen_plotly_scatter(fig, df_sel['每坪單價(萬)'], df_sel['模型預估(萬/坪)'], margin=margin,
                                                color=color_set, text=hover_text, opacity=1,
                                                xlabel='實際單價(萬/坪)', ylabel='預估單價(萬/坪)', title=title)

                st.write('')
                st.plotly_chart(fig, config=config, theme=None)

                if is_rf:
                    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 1, "colspan": 2}, None], [{}, {}]],
                                        subplot_titles=('信心指標 v.s. 絕對誤差', '信心分佈', '誤差(萬/坪)分佈'))

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
                        fig.add_vrect(x0=-1 * err_max, row=2, col=2, x1=err_max, line_width=0, fillcolor="red",
                                      opacity=0.1)
                        fig.add_vrect(x0=th_l, row=2, col=1, x1=th_h, line_width=0, fillcolor="red", opacity=0.1)
                        fig.add_vrect(x0=th_l, row=1, col=1, x1=th_h, line_width=0, fillcolor="red", opacity=0.1)

                    st.plotly_chart(fig, config=config, theme=None)

            st.write('')
            AgGrid(df_show, theme='blue', enable_enterprise_modules=True)

            del df
        else:
            st.write(f'此檔共有 {n_data}筆 資料, 經篩選後不可進行模型預估 !')


@fn_profiler
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

    f_num = ['屋齡', '交易年', '交易月']
    f_num += ['台北市', '緯度', '經度']
    f_num += ['建物坪數', '車位坪數', '幾房', '幾廳', '幾衛']
    f_num += ['總樓層數', '頂樓', '移轉層次']
    f_num += ['sku_dist', 'sku_total']
    f_num += ['MRT_DIST', 'MRT_Tput_UL', 'MRT_Tput_DL', 'MRT_Tput', 'MRT_Commute_Time_UL']
    f_num += ['利率_13個月前', '利率_15個月前']
    f_num += ['頂樓-1']
    f_num += ['使用分區_住']
    f_num += ['MRT_ave', 'SKU_ave', 'DIST_ave']
    f_num += ['稅_中位數', '稅_平均數', '稅_第一分位數', '稅_第三分位數']

    f_num += ['地下樓層', '總戶數', '基地面積(坪)']

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
        if X[c].dtype == object and is_inference:
            print(c, X[c].dtype, 'change typ to float !')
            X[c] = X[c].astype(float)
        if X[c].isna().any() and is_inference:
            print(c, X[[c]].shape, X.shape)
            for i, v in enumerate(X[c].tolist()):
                print(c, i, v, df_cat.iloc[i, :].values)
            assert False, c + str(f' is {X[c].values} {len(X[c].tolist())}')

    # print(X.shape, df_cat.shape)
    return X, df_cat


@fn_profiler
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


def fn_gen_plotly_hist(fig, data, title, row=1, col=1, margin=None, bins=100, line_color='white',
                       showlegend=False, lg=None, hovertext=None, barmode='stack', opacity=0.8, color=None,
                       width=None, height=None, font_size=None):
    fig.add_trace(
        go.Histogram(x=data, name=title, legendgroup=lg, showlegend=showlegend, nbinsx=bins, hovertext=hovertext,
                     marker=dict(
                         opacity=opacity,
                         color=color,
                         line=dict(
                             color=line_color, width=0.4
                         ),
                     )),
        row=row,
        col=col,
    )

    fig.update_layout(margin=margin,
                      barmode=barmode,
                      width=width,
                      height=height,
                      font=dict(
                          family=None,  # "Courier New, monospace",
                          size=font_size,
                          color=None, )
                      )

    return fig


def fn_gen_plotly_bar(df_top, x_data_col, y_data_col, text_col, v_or_h, margin,
                      color_col=None, text_fmt=None, title=None, x_title=None, y_title=None, ccs='agsunset', op=None,
                      height=None, width=None, font_size=None):
    fig = px.bar(df_top, x=x_data_col, y=y_data_col,
                 orientation=v_or_h, title=title,
                 text=text_col, color=color_col,
                 color_continuous_scale=ccs,
                 opacity=op)

    fig.update_traces(texttemplate=text_fmt)
    fig.update_layout(margin=margin,
                      yaxis_title=y_title,
                      xaxis_title=x_title,
                      height=height,
                      font=dict(
                          family=None,
                          size=font_size,
                          color=None, )
                      )

    return fig


def fn_gen_plotly_map(df, title, hover_name, hover_data, map_style,
                      color=None, zoom=10, height=400, text=None, margin=None, op=None, size=None):
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

    # color_mid = np.average(df['每坪單價(萬)'])
    fig = px.scatter_mapbox(df,
                            lat=lat, lon=lon, title=title,
                            hover_name=hover_name, hover_data=hover_data,
                            color_discrete_sequence=["fuchsia"],
                            color_continuous_scale='portland',  # jet
                            # color_continuous_midpoint=color_mid,
                            zoom=zoom, height=height, color=color,
                            text=text, opacity=op, size=size)

    fig.update_layout(mapbox_style=map_style, margin=margin, width=800, height=450)  # 'mapbox_style=map_style'
    # map style - "open-street-map", "white-bg", "carto-positron", "stamen-terrain"

    return fig


def fn_gen_plotly_scatter(fig, x_data, y_data, row=1, col=1, margin=None, color=None, text=None, op=None, opacity=0.3,
                          xlabel=None, ylabel=None, title=None, size=None, marker_sym=None,
                          legend=False, name=None, update_layout=True, line_color=None, line_width=None, mode='markers',
                          colorscale='Bluered', width=None, height=None):
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode=mode, showlegend=legend, hovertext=text,
                             marker_symbol=marker_sym, name=name,
                             line=dict(color=line_color, width=line_width),
                             opacity=op,
                             marker=dict(
                                 size=size,
                                 opacity=opacity,
                                 # line=dict(color=line_color, width=line_width),  # {'color': line_color, 'width': line_width},
                                 color=color,
                                 colorscale=colorscale)  # "Viridis" portland Bluered
                             ), row=row, col=col)

    # if margin is not None:
    # fig.update_layout(margin=margin, width=width)

    if update_layout:
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
            font=dict(size=13),
            width=width,
            height=height,
            margin=margin,
        )

    return fig


def fn_gen_plotly_treemap(df, path=None, values=None, color=None, hover=None, mid=None):
    fig = px.treemap(df, path=path, values=values,
                     color=color, hover_data=hover,
                     color_continuous_scale='balance',  # balance RdBu
                     color_continuous_midpoint=mid)

    fig.update_layout(margin=dict(t=20, l=0, r=0, b=20), width=850, height=500)

    return fig


def fn_gen_df_color(val):
    color = 'greenyellow' if val else 'lightgray'

    return f'background-color: fcolor]'


def fn_add_date_line(fig, df, date, mode='lines', width=10, color='lightgreen', dash=None, op=None):
    fig.add_trace(
        go.Scatter(
            x=[date, date],
            y=[df.iloc[0, 0], df.iloc[-1, 0]],
            mode=mode,
            line=go.scatter.Line(color=color, width=width, dash=dash),
            showlegend=False,
            opacity=op,
        )
    )
    return fig


@fn_profiler
def fn_gen_analysis_admin(df, margin=None, bc_name=None):
    color_by = '無'
    c1, c2, c3 = st.columns(3)
    dists = ['不限'] + list(df['鄉鎮市區'].unique())

    if bc_name is None or '不限' in bc_name:
        dist_of_bc = '北投區' if '北投區' in dists else dists[-1]
    else:
        dist_of_bc = df[df['建案名稱'] == bc_name[0]]['鄉鎮市區'].values[0]

    dist = c1.selectbox('行政區', options=dists, index=dists.index(dist_of_bc))
    tax = c2.selectbox('各里所得分析(108年度)', options=['無', '所得中位數', '所得平均數', '全選'], index=1)
    op = 0.2  # c3.slider('透明度', min_value=0.01, max_value=0.4, value=0.2)
    st.write('')

    # if bc_name is None:
    #     bc_name = ['康寶日出印象']
    margin = {'l': 0, 'r': 30, 't': 30, 'b': 20} if margin is None else margin
    admin_dists = len(df['鄉鎮市區'].unique())

    df_dist = df.copy() if dist == '不限' else df[df['鄉鎮市區'] == dist]

    d_v = df_dist.loc[:, '鄉鎮市區'] + '_' + df_dist.loc[:, '里']
    df_dist.loc[:, '里'] = d_v.copy()

    df_dist = pd.DataFrame(df_dist.groupby('里', as_index=True)['每坪單價(萬)'].mean())
    df_dist = df_dist[['每坪單價(萬)']].apply(lambda x: round(x, 2))
    df_dist.reset_index(inplace=True)
    df_dist.rename(columns={'index': '里'})
    admin_vills = len(df_dist['里'].unique())
    dist_sel = dist.replace("不限", "台北市")

    # fig_sct = make_subplots(rows=2, cols=1,
    #                         # specs=[[{"rowspan": 2, "colspan": 1}, None], [{}, {}], [{}, {}]],
    #                         subplot_titles=(f'台北市 {admin_dists}個 行政區 v.s. 每坪單價(萬)',
    #                                         f'{dist_sel} {admin_vills}個 里 v.s. 每坪單價(萬)'))

    df_sort = df.sort_values(by='DIST_ave', ascending=False)
    df_gb = pd.DataFrame(df_sort.groupby('鄉鎮市區', as_index=True)['每坪單價(萬)'].mean())
    df_gb = df_gb[['每坪單價(萬)']].apply(lambda x: round(x, 2))
    df_gb = df_gb.sort_values(by='每坪單價(萬)', ascending=False)

    if bc_name != '不限':
        df_hl = df_sort[df_sort['建案名稱'].apply(lambda x: x in bc_name)]

    # color_set, opacity = fn_set_color_by(color_by, df_sort)

    fig_sct = make_subplots(rows=2, cols=1,
                            # specs=[[{"rowspan": 2, "colspan": 1}, None], [{}, {}], [{}, {}]],
                            subplot_titles=(f'台北市 {admin_dists}個 行政區 v.s. 每坪單價(萬)',
                                            f'{dist_sel} {admin_vills}個 里 v.s. 每坪單價(萬)'))

    # Jack Force Order
    hover_text = fn_get_hover_text(df_gb)
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_gb.index, df_gb['每坪單價(萬)'].values,
                                    margin=margin, color='tomato', text=hover_text,
                                    opacity=0.8, row=1, size=15, marker_sym=24,
                                    legend=True, name='每坪均價(區)', width=700)

    hover_text = fn_get_hover_text(df_sort)
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['鄉鎮市區'], df_sort['每坪單價(萬)'],
                                    margin=margin, color='royalblue', text=hover_text, opacity=op, row=1)

    df_dist_hl = df_sort if dist == '不限' else df_sort[df_sort['鄉鎮市區'] == dist]
    hover_text = fn_get_hover_text(df_dist_hl)
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_dist_hl['鄉鎮市區'], df_dist_hl['每坪單價(萬)'],
                                    margin=margin, color='lightseagreen', text=hover_text, opacity=0.8, row=1, size=8,
                                    width=700)

    if bc_name != '不限':
        hover_txt1 = fn_get_hover_text(df_hl)

        fig_sct = fn_gen_plotly_scatter(fig_sct, df_hl['鄉鎮市區'], df_hl['每坪單價(萬)'],
                                        margin=margin, color='red', text=hover_txt1, opacity=1, row=1, size=8)

    # hover_text = fn_get_hover_text(df_gb)
    # fig_sct = fn_gen_plotly_scatter(fig_sct, df_gb.index, df_gb['每坪單價(萬)'].values,
    #                                 margin=margin, color='tomato', text=hover_text,
    #                                 opacity=0.6, row=1, size=12, marker_sym=24,
    #                                 legend=True, name='每坪均價(區)')

    df_sort = df_dist.sort_values(by='每坪單價(萬)', ascending=False)

    df_vill = pd.DataFrame()
    df['dist_vill'] = df['鄉鎮市區'] + '_' + df['里']
    for vill in df_sort['里'].values:
        df_vill = pd.concat([df_vill, df[df['dist_vill'] == vill]], axis=0)

    hover_text = fn_get_hover_text(df_sort)
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['里'], df_sort['每坪單價(萬)'],
                                    margin=margin, color='violet', text=hover_text,
                                    opacity=0.8, row=2, size=15, marker_sym=24,
                                    legend=True, name='每坪均價(里)', width=700)

    hover_text = fn_get_hover_text(df_vill)
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_vill['dist_vill'], df_vill['每坪單價(萬)'],
                                    margin=margin, color='lightseagreen', text=hover_text, opacity=min(1., op * 3),
                                    row=2, width=700)

    # hover_text = fn_get_hover_text(df_sort)
    # fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['里'], df_sort['每坪單價(萬)'],
    #                                 margin=margin, color='violet', text=hover_text,
    #                                 opacity=0.6, row=2, size=12, marker_sym=24,
    #                                 legend=True, name='每坪均價(里)')

    if tax == '所得平均數' or tax == '全選':
        df_tax_ave = pd.DataFrame(df_sort['里'].apply(lambda x: df[df['區_里'] == x]['稅_平均數'].values[0] / 10))
        df_tax_ave.rename(columns={'里': '稅_平均數(萬)'}, inplace=True)
        hover_text = fn_get_hover_text(df_tax_ave)
        fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['里'], df_tax_ave['稅_平均數(萬)'],
                                        margin=margin, color='tomato', text=hover_text,
                                        opacity=0.7, row=2, size=11, marker_sym=3,
                                        legend=True, name='所得平均', height=600, width=700)

    if tax == '所得中位數' or tax == '全選':
        df_tax_med = pd.DataFrame(df_sort['里'].apply(lambda x: df[df['區_里'] == x]['稅_中位數'].values[0] / 10))
        df_tax_med.rename(columns={'里': '稅_中位數(萬)'}, inplace=True)
        hover_text = fn_get_hover_text(df_tax_med)
        fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['里'], df_tax_med['稅_中位數(萬)'],
                                        margin=margin, color='orange', text=hover_text,
                                        opacity=0.7, row=2, size=11, marker_sym=17,
                                        legend=True, name='所得中位數', height=600, width=700)

    if tax in ['全選', '所得中位數', '所得平均數']:
        fig_sct_2 = make_subplots(rows=2, cols=1,
                                  subplot_titles=(f'😣 購屋痛苦指數 ({dist_sel}各里 年所得中位數 - 每坪均價)',
                                                  f'😣 購屋痛苦指數 ({dist_sel}各里 年所得平均數 - 每坪均價)'))

        df_1 = df_sort
        if tax in ['全選', '所得中位數']:
            df_1['中位數_均價'] = df_tax_med['稅_中位數(萬)'] - df_sort['每坪單價(萬)']
            df_1 = df_1.sort_values(by='中位數_均價', ascending=False)
            hover_text = fn_get_hover_text(df_1)

            fig_sct_2 = fn_gen_plotly_scatter(fig_sct_2, df_1['里'], df_1['中位數_均價'],
                                              margin=margin, color='red', text=hover_text,
                                              opacity=1, row=1, size=12, marker_sym=4,  # marker_sym=18,
                                              legend=True, name='入不敷出', height=600, width=700)

            df_1_ok = df_1[df_1['中位數_均價'] >= 0]
            fig_sct_2 = fn_gen_plotly_scatter(fig_sct_2, df_1_ok['里'], df_1_ok['中位數_均價'],
                                              margin=margin, color='lightseagreen', text=hover_text,
                                              opacity=1, row=1, size=15, marker_sym=0,  # marker_sym=18,
                                              legend=True, name='入可敷出', height=600, width=700)

        if tax in ['全選', '所得平均數']:
            df_1['平均數_均價'] = df_tax_ave['稅_平均數(萬)'] - df_sort['每坪單價(萬)']
            df_1 = df_1.sort_values(by='平均數_均價', ascending=False)
            fig_sct_2 = fn_gen_plotly_scatter(fig_sct_2, df_1['里'], df_1['平均數_均價'],
                                              margin=margin, color='red', text=hover_text,
                                              opacity=1, row=2, size=12, marker_sym=4,
                                              legend=True, name='入不敷出', height=600)

            df_1_ok = df_1[df_1['平均數_均價'] >= 0]
            fig_sct_2 = fn_gen_plotly_scatter(fig_sct_2, df_1_ok['里'], df_1_ok['平均數_均價'],
                                              margin=margin, color='lightseagreen', text=hover_text,
                                              opacity=1, row=2, size=15, marker_sym=0,
                                              legend=True, name='入可敷出', height=600)

        return [fig_sct, fig_sct_2]
    else:
        return [fig_sct]


@fn_profiler
def fn_gen_analysis_mrt(df, color_by, margin=None, bc_name=None):
    # if bc_name is None:
    #     bc_name = ['康寶日出印象']

    # dist_of_bc = '不限' if bc_name is None or bc_name == '不限' else df[df['建案名稱'] == bc_name[0]]['鄉鎮市區'].values[0]
    # df = df[df['鄉鎮市區'] == dist_of_bc] if dist_of_bc != '不限' else df

    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin
    mrts = len(df['MRT'].unique())

    fig_sct = make_subplots(rows=3, cols=2,
                            specs=[[{"colspan": 2, "rowspan": 2}, None], [{}, {}], [{}, {}]],
                            subplot_titles=(f'{mrts}個 鄰近捷運站 V.S. 每坪單價(萬)',))
    # df_sort = df.sort_values(by='每坪單價(萬),ascending=True)

    df_ave = pd.DataFrame(df.groupby(['MRT'])['每坪單價(萬)'].mean())

    df['MRT_ave'] = df['MRT'].apply(lambda x: round(df_ave.loc[x, '每坪單價(萬)'], 2))

    df_sort = df.sort_values(by='MRT_ave', ascending=False)
    df_hl = df_sort[df_sort['建案名稱'].apply(lambda x: x in bc_name)]

    hover_text = fn_get_hover_text(df_sort)

    color_set, opacity = fn_set_color_by(color_by, df_sort)

    fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort['MRT'], df_sort['每坪單價(萬)'],
                                    margin=margin, color=color_set, text=hover_text)

    df_sort_ave = df_sort.drop_duplicates(subset=['MRT'], keep='first')
    fig_sct = fn_gen_plotly_scatter(fig_sct, df_sort_ave['MRT'], df_sort_ave['MRT_ave'], row=1, col=1, margin=margin,
                                    color='violet', opacity=0.7, marker_sym=24, size=13)

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


@fn_profiler
def fn_gen_analysis_sku(df, color_by, margin=None, bc_name=None):
    # if bc_name is None:
    #     bc_name = ['康寶日出印象']
    SKUs = len(df['sku_name'].unique())
    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin

    df_ave = pd.DataFrame(df.groupby(['sku_name'])['每坪單價(萬)'].mean())

    df['SKU_ave'] = df['sku_name'].apply(lambda x: round(df_ave.loc[x, '每坪單價(萬)'], 2))

    df_sort = df.sort_values(by='SKU_ave', ascending=False)

    df_sort['sku_name'] = df_sort['sku_name'].apply(
        lambda x: x.replace('高中', '').replace('中學', '').replace('實驗', '').replace('國立', ''))

    df_hl = df_sort[df_sort['建案名稱'].apply(lambda x: x in bc_name)]
    color_set, opacity = fn_set_color_by(color_by, df_sort)

    y_data = df_sort['每坪單價(萬)']

    hover_text = fn_get_hover_text(df_sort)

    fig_sku_1 = make_subplots(rows=3, cols=2,
                              specs=[[{"rowspan": 2, "colspan": 2}, None], [{}, {}], [{}, {}]],
                              subplot_titles=(f'{SKUs}個鄰近小學 v.s.每坪單價(萬)',))
    fig_sku_1 = fn_gen_plotly_scatter(fig_sku_1, df_sort['sku_name'], y_data, row=1, col=1, margin=margin,
                                      color=color_set, text=hover_text, opacity=0.5)

    df_sort_ave = df_sort.drop_duplicates(subset=['sku_name'], keep='first')
    fig_sku_1 = fn_gen_plotly_scatter(fig_sku_1, df_sort_ave['sku_name'], df_sort_ave['SKU_ave'], row=1, col=1,
                                      margin=margin,
                                      color='violet', opacity=0.7, marker_sym=24, size=13)

    hover_txt1 = fn_get_hover_text(df_hl)
    fig_sku_1 = fn_gen_plotly_scatter(fig_sku_1, df_hl['sku_name'], df_hl['每坪單價(萬)'], row=1, col=1, margin=margin,
                                      color='red', text=hover_txt1, opacity=1)

    fig_sku_2 = make_subplots(rows=2, cols=2,
                              specs=[[{}, {}], [{}, {}]],
                              subplot_titles=('鄰近小學距離(公尺) v.s. 每坪單價(萬)',
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


@fn_profiler
def fn_gen_analysis_building(df, target, color_by, margin=None, bc_name=None):
    # if bc_name is None:
    #     bc_name = ['康寶日出印象']

    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin
    y_data = df[target]

    color_set, opacity = fn_set_color_by(color_by, df)

    # hover_text = df['鄉鎮市區'] + ', ' + \
    #              df['建案名稱'].astype(str) + ', ' + \
    #              df['交易年'].astype(str) + '年, ' + \
    #              df['建物坪數'].astype(int).astype(str) + '坪, ' + \
    #              df['總樓層數'].astype(int).astype(str) + '樓'

    hover_text = fn_get_hover_text(df)
    df_hl = df if bc_name is None else df[df['建案名稱'].apply(lambda x: x in bc_name)]  # <--

    fig_sct_3 = make_subplots(rows=2, cols=2,
                              subplot_titles=(f'交易年 v.s. {target}', f'建物坪數 v.s. {target}',
                                              f'移轉層次 v.s {target}', f'總樓層數 v.s. {target}'))

    fig_sct_3 = fn_gen_plotly_scatter(fig_sct_3, df['交易年'], y_data, row=1, col=1, margin=margin, color=color_set,
                                      text=hover_text, opacity=opacity)

    # hover_txt1 = df_hl['鄉鎮市區'] + ',' + \
    #              df_hl['建案名稱'].astype(str) + ',' + \
    #              df_hl['交易年'].astype(str) + '年,' + \
    #              df_hl['建物坪數'].astype(int).astype(str) + '坪,' + \
    #              df_hl['總樓層數'].astype(int).astype(str) + '樓'

    hover_txt1 = fn_get_hover_text(df_hl)

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


def fn_gen_hist_analysis(fig, df_list, cols, colors, years, yr, margin, lg=None):
    bins = 30 if cols[0] == '交易年' else 50

    if len(df_list) == 1:
        df1, df2, df3, df4 = df_list[0], df_list[0], df_list[0], df_list[0]
    else:
        df1, df2, df3, df4 = df_list[0], df_list[1], df_list[2], df_list[3]

    df1 = df1[df1['交易年'] == yr]
    df2 = df2[df2['交易年'] == yr]
    df3 = df3[df3['交易年'] == yr]
    df4 = df4[df4['交易年'] == yr]

    fig = fn_gen_plotly_hist(fig, df1[cols[0]], f'{yr}年', row=1, col=1, bins=bins, margin=margin,
                             showlegend=False, lg=lg,
                             barmode='stack', color=colors[years.index(yr)])

    fig = fn_gen_plotly_hist(fig, df2[cols[1]], f'{yr}年', row=1, col=2, bins=bins, margin=margin,
                             showlegend=True, lg=lg,
                             barmode='stack', color=colors[years.index(yr)])

    fig = fn_gen_plotly_hist(fig, df3[cols[2]], f'{yr}年', row=2, col=1, bins=bins, margin=margin,
                             showlegend=False, lg=lg,
                             barmode='stack', color=colors[years.index(yr)])

    fig = fn_gen_plotly_hist(fig, df4[cols[3]], f'{yr}年', row=2, col=2, bins=bins, margin=margin,
                             showlegend=False, lg=lg,
                             barmode='stack', color=colors[years.index(yr)])

    return fig


@fn_profiler
def fn_gen_analysis_statistic(df):
    df = df[df['地下樓層'].apply(lambda x: str(x).isnumeric())]
    df['地下樓層'] = df['地下樓層'].astype(int)

    df = df[df['容積率(%)'].apply(lambda x: str(x).isnumeric())]
    df['容積率(%)'] = df['容積率(%)'].astype(int)

    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20}

    fig_bar = make_subplots(rows=2, cols=2, subplot_titles=('交易量/年', '交易量/月', '每坪單價(萬)', '總價(萬)'))
    fig_bar_2 = make_subplots(rows=2, cols=2, subplot_titles=('建物坪數', '總樓層數', '車位類別', '車位單價(萬)'))
    df_pk_1 = df[df['車位類別'] == '坡道平面']
    df_pk_2 = df[df['車位類別'] == '坡道機械']
    fig_bar_3 = make_subplots(rows=2, cols=2,
                              subplot_titles=('坡道平面 的 價格分佈', '坡道平面 的 坪數分佈', '坡道機械 的 價格分佈', '坡道機械 的 坪數分佈'))

    dists = len(df['鄉鎮市區'].unique())
    df_typ = df[df['都市土地使用分區'].apply(lambda x: x == '住' or x == '商')]
    fig_bar_4 = make_subplots(rows=2, cols=2, subplot_titles=(f'土地使用分區', f'行政區({dists}個)', '地下樓層', '總戶數'))
    fig_bar_5 = make_subplots(rows=2, cols=2, subplot_titles=(f'基地面積(坪)', f'容積率(%)', '建蔽率(%)', '公設比(%)'))

    df['交易年'] = df['交易年'].astype(int)
    years = list(df['交易年'].unique())
    colors = plotly.colors.qualitative.Vivid
    for yr in years:
        fig_bar = fn_gen_hist_analysis(fig_bar, [df], ['交易年', '交易月', '每坪單價(萬)', '總價(萬)'], colors, years, yr,
                                       margin, lg=str(yr))

        fig_bar_2 = fn_gen_hist_analysis(fig_bar_2, [df], ['建物坪數', '總樓層數', '車位類別', '車位單價(萬)'], colors, years, yr,
                                         margin, lg=str(yr))

        fig_bar_3 = fn_gen_hist_analysis(fig_bar_3, [df_pk_1, df_pk_1, df_pk_2, df_pk_2],
                                         ['車位單價(萬)', '車位坪數', '車位單價(萬)', '車位坪數'], colors, years, yr,
                                         margin, lg=str(yr))

        fig_bar_4 = fn_gen_hist_analysis(fig_bar_4, [df_typ, df, df, df],
                                         ['都市土地使用分區', '鄉鎮市區', '地下樓層', '總戶數'], colors, years, yr,
                                         margin, lg=str(yr))

        fig_bar_5 = fn_gen_hist_analysis(fig_bar_5, [df_typ, df, df, df],
                                         ['基地面積(坪)', '容積率(%)', '建蔽率(%)', '公設比(%)'], colors, years, yr,
                                         margin, lg=str(yr))

    return fig_bar, fig_bar_2, fig_bar_3, fig_bar_4, fig_bar_5


def fn_gen_analysis_sel(df, build_case, latest_records, key='k', colors=None,
                        sel_option=['dist', 'build_case', 'color_by'], dist_default=None):
    bc = 'NA'
    color_by = 'NA'

    c1, c2, c3 = st.columns(3)
    dists = ['不限'] + list(df['鄉鎮市區'].unique())
    # dist_dft = dists.index('北投區') if '北投區' in dists else 0

    if build_case is not None and build_case != '不限':
        df_bc = df[df['建案名稱'] == build_case]
        if dist_default == '不限':
            pass
        else:
            dist_dft = df_bc.loc[:, '鄉鎮市區'].values[0]
            dist_dft = dists.index(dist_dft)

    dist = c1.selectbox('行政區', options=dists, index=dist_dft, key=f'{key}+dist')
    # dist = c1.selectbox('行政區', options=dists, index=0, key=f'{key}+dist')
    df = df if dist == '不限' else df[df['鄉鎮市區'] == dist]

    # df.sort_values(by='coor_ave', ascending=False, inplace=True)
    build_cases = ['不限'] + [b for b in list(df['建案名稱'].unique()) if str(b) != 'nan']
    bc_idx = build_cases.index(build_case) if build_case in build_cases else len(build_cases) - 1

    if 'build_case' in sel_option:
        bc = c2.selectbox(f'建案(共{len(build_cases) - 1}個)', options=build_cases, index=bc_idx, key=f'{key}+bc')
    else:
        bc = '不限'

    bc = bc.split('_')[0]

    if 'color_by' in sel_option:
        colors = ['無', '依交易年', '依總樓層數', '依建物坪數', f'依最新登({latest_records})'] if colors == None else colors
        color_by = c3.selectbox('著色條件', options=colors, index=0, key=f'{key}+color')

    return df, bc, color_by, [c1, c2, c3]


@fn_profiler
def fn_gen_analysis_sale_period(df, bc, margin=None, op=0.8):
    df['date'] = df['交易年月日'].apply(lambda x: str(int(x) + 19110000))
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date
    dists = list(df['鄉鎮市區'].unique())
    dist = dists[0] if len(dists) == 1 else '台北市'

    r = st.radio('排序方式:', ['依最早交易', '依銷售量', '依銷售速率(銷量/月)', '依銷售週期(月)', '依銷售總額'], index=0)
    fn_set_radio_2_hor()

    df_bc_s = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['date'].min()).rename(columns={'date': '最早'})
    df_bc_e = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['date'].max()).rename(columns={'date': '最新'})
    df_bc_c = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['date'].count()).rename(columns={'date': '銷量'})
    df_bc_t = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['總價(萬)'].sum()).rename(columns={'總價(萬)': '總額(億)'})
    df_bc_d = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['鄉鎮市區'].min()).rename(columns={'鄉鎮市區': '行政區'})
    df_bc_v = pd.DataFrame(df.groupby(['建案名稱'], as_index=True)['里'].min())

    df_bc = pd.concat([df_bc_s, df_bc_e, df_bc_c, df_bc_t, df_bc_d, df_bc_v], axis=1)
    df_bc['總額(億)'] = df_bc['總額(億)'].apply(lambda x: round(x / 10000, 2))

    df_bc.reset_index(inplace=True)
    df_bc.rename(columns={'建案名稱': '建案'}, inplace=True)

    fr, to = df['date'].min(), df['date'].max()
    fr_dft = fr if bc == '不限' else df_bc[df_bc['建案'] == bc]['最早'].values[0]
    to_dft = to if bc == '不限' else df_bc[df_bc['建案'] == bc]['最新'].values[0]

    with st.form(key='sale1'):
        period = st.slider('選擇 觀察週期 (西元 年-月)', min_value=fr, max_value=to, value=(fr_dft, to_dft),
                           step=datetime.timedelta(days=31), format='YY-MM')
        submitted = st.form_submit_button('設定')
        if submitted:
            fr_dft, to_dft = period[0], period[1]

    df_bc = df_bc[df_bc['最新'] >= fr_dft]
    df_bc = df_bc[df_bc['最早'] <= to_dft]

    for idx in df_bc.index:
        s = df_bc.loc[idx, '最早']
        e = df_bc.loc[idx, '最新']
        df_bc.at[idx, '週期'] = 12 * (e.year - s.year) + e.month - s.month + 1
        df_bc.at[idx, '銷售速率'] = round(df_bc.at[idx, '銷量'] / df_bc.at[idx, '週期'], 1)

    if r == '依銷售量':
        df_bc.sort_values(by='銷量', inplace=True, ascending=False)
        color = '銷量'
    elif r == '依最早交易':
        df_bc.sort_values(by='最早', inplace=True, ascending=True)
        color = '銷量'
    elif r == '依銷售週期(月)':
        df_bc.sort_values(by='週期', inplace=True, ascending=False)
        color = '銷量'
    elif r == '依銷售速率(銷量/月)':
        df_bc.sort_values(by='銷售速率', inplace=True, ascending=False)
        color = '銷售速率'
    elif r == '依銷售總額':
        df_bc.sort_values(by='總額(億)', inplace=True, ascending=False)
        color = '總額(億)'
    else:
        color = None

    title = f'{fr_dft.year}.{fr_dft.month}~{to_dft.year}.{to_dft.month}, ' \
            f'{12 * (to_dft.year - fr_dft.year) + (to_dft.month - fr_dft.month + 1)}個月 {dist} {df_bc.shape[0]}個建案'

    margin = {'l': 0, 'r': 50, 't': 30, 'b': 20} if margin is None else margin
    fig = px.timeline(df_bc, x_start='最早', x_end='最新', y='建案', color=color,
                      hover_data=['銷售速率', '銷量', '週期', '總額(億)', '行政區', '里'],
                      color_continuous_scale='portland', opacity=op)
    fig.update_yaxes(autorange="reversed", title={'text': ''})
    fig.update_xaxes(tickformat="%Y-%m")
    fig.update_layout(margin=margin,
                      title={
                          'text': f'{title}',
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'
                      }, )

    fig = fn_add_date_line(fig, df_bc, datetime.date.today())
    fig = fn_add_date_line(fig, df_bc, fr_dft, dash='dot', color='orangered', width=3, op=0.5)
    fig = fn_add_date_line(fig, df_bc, to_dft, dash='dot', color='orangered', width=3, op=0.5)

    df = df[df['date'] >= fr_dft]
    df = df[df['date'] <= to_dft]

    df['Y_M'] = df['date'].apply(lambda x: datetime.date(x.year, x.month, 1))

    df_ym = pd.DataFrame(df.groupby(['Y_M'], as_index=False)['總價(萬)'].sum())
    df_ym['總價(億)'] = df_ym['總價(萬)'].apply(lambda x: round(x / 10000, 1))

    df_area = pd.DataFrame(df.groupby(['Y_M'], as_index=False)['建物坪數'].sum())
    df_area['銷售面積(百坪)'] = df_area['建物坪數'].apply(lambda x: round(x / 100, 1))

    df_area['均價'] = df_ym['總價(萬)'] / df_area['建物坪數']
    df_area['均價'] = df_area['均價'].apply(lambda x: round(x, 2))

    fig_bar = go.Figure(data=[
        go.Bar(x=df_ym['Y_M'], y=df_ym['總價(億)'], name='銷售總額(億)', opacity=op),
        go.Line(x=df_area['Y_M'], y=df_area['銷售面積(百坪)'], name='銷售面積(百坪)', mode='lines+markers'),
        go.Line(x=df_area['Y_M'], y=df_area['均價'], name='每坪均價(萬/坪)', mode='lines+markers'),
    ])

    price_all = int(df_ym['總價(億)'].sum())
    fig_bar.update_layout(title_text=f'{title} 銷售總額{price_all}億',
                          title_x=0.5,
                          margin=dict(l=100, r=10, t=30, b=40))

    fig_bar.update_xaxes(tickformat="%Y-%m")

    return fig, fig_bar


def fn_corr_util(dic_of_show, df_1, cmp, target, config):
    d = dic_of_show[cmp]
    hovertext = fn_get_hover_text(df_1)

    fig = go.Figure(data=go.Splom(
        dimensions=d + target,
        diagonal=dict(visible=False),
        hovertext=hovertext,
        showupperhalf=False,
        showlowerhalf=True,
        marker=dict(color=df_1['每坪單價(萬)'],
                    size=6,
                    # colorscale='Bluered',
                    # colorscale='Bluered',
                    colorscale='Bluered',
                    line=dict(width=0.5,
                              color='rgb(230,230,230)'))))

    title = f'每坪單價 與 "{cmp}" 指標 的關係'
    fig.update_layout(title=title,
                      dragmode='select',
                      width=800,
                      height=800,
                      hovermode='closest')

    st.plotly_chart(fig, config=config)


def fn_plot_shp(fig, dic_of_shp_data, k, text=None, row=None, col=None):
    assert k in dic_of_shp_data['shape'].keys(), f'{k} not in dic_of_shp[shape].keys('
    x, y = dic_of_shp_data['shape'][k].exterior.xy
    fig = fn_gen_plotly_scatter(fig, list(x), list(y),
                                row=row, col=col,
                                # color='red', opacity=0.8, size=4,
                                text=text, opacity=1,
                                update_layout=False,
                                line_color='blue', line_width=3,
                                mode='lines', colorscale=None)

    return fig


# @fn_profiler
def fn_gen_analysis(df, latest_records, build_case):
    config = {'scrollZoom': True,
              'toImageButtonOptions': {'height': None, 'width': None}}

    tabs = st.tabs(['單價分布', '特徵分布', '相關分析', '行政區分析', '捷運分析', '小學分析', '建物分析', '銷售分析'])
    tab_dist_price, tab_dist_char, tab_ana_corr, tab_ana_dist, tab_ana_mrt, tab_ana_ele, tab_ana_bd, tab_ana_sell = tabs
    fn_dbg('fn_gen_web_eda 3-1-1')

    with tab_dist_price:  # st.expander(f'👓 檢視 每坪單價 的 分布狀況'):
        df_1, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='pr',
                                                             sel_option=['dist'], dist_default=None)

        bc_show = c[1].selectbox('建案位置', options=['建案位置: 隱藏', '建案位置: 顯示'], index=1, key='show_bc')

        if build_case_sel == '不限':
            bc_vill = build_case_sel
            bc_dist = build_case_sel
        else:
            bc_vill = df_1[df_1['建案名稱'] == build_case_sel]['里'].values[0]
            bc_dist = df_1[df_1['建案名稱'] == build_case_sel]['鄉鎮市區'].values[0]

        fig_3d = px.scatter_3d(df_1, x='經度', y='緯度', z='每坪單價(萬)', color='每坪單價(萬)',
                               hover_data=['鄉鎮市區', '建案名稱', '交易年', 'MRT', 'sku_name'],
                               opacity=0.8, color_continuous_scale='portland')
        fig_3d.update_layout(title='每坪單價 的 分佈狀況', autosize=True,
                             width=700, height=500,
                             margin={'l': 0, 'r': 0, 't': 30, 'b': 20})

        fig_c = go.Figure(
            data=go.Contour(x=df_1['經度'], y=df_1['緯度'], z=df_1['coor_ave'], line_smoothing=1.2, colorscale='portland',
                            contours=dict(
                                showlabels=True,  # show labels on contours
                                labelfont=dict(  # label font properties
                                    size=12,
                                    color='white',
                                )
                            )))

        dist = '不限' if len(df_1['鄉鎮市區'].unique()) > 1 else df_1['鄉鎮市區'].values[0]

        if dist == '不限':
            for k in dic_of_shp_dist['shape'].keys():
                vill = dic_of_shp_dist['properties'][k]
                # fn_dbg(f'{k} {dist}, {vill}, {dist in vill}')
                x, y = dic_of_shp_dist['shape'][k].exterior.xy
                fig_c = fn_gen_plotly_scatter(fig_c, list(x), list(y),
                                              row=None, col=None,
                                              # color='white', size=4,
                                              text=vill, opacity=1,
                                              update_layout=False,
                                              line_color='white', line_width=1.5,
                                              mode='lines')

            if bc_dist.endswith('區'):
                k = f'{"臺北市"}, {bc_dist}, {"NA"}'
                fig_c = fn_plot_shp(fig_c, dic_of_shp_dist, k, text=k)

                # assert k in dic_of_shp_dist['shape'].keys(), f'{k} not in dic_of_shp[shape].keys('
                # x, y = dic_of_shp_dist['shape'][k].exterior.xy
                # fig_c = fn_gen_plotly_scatter(fig_c, list(x), list(y),
                #                               row=None, col=None,
                #                               # color='red', opacity=0.8, size=4,
                #                               text=vill, opacity=1,
                #                               update_layout=False,
                #                               line_color='blue', line_width=3,
                #                               mode='lines', colorscale=None)

        else:  # dist in vill:

            fn_dbg("fn_gen_web_eda 3-1-1-1")

            for k in dic_of_shp['shape'].keys():
                vill = dic_of_shp['properties'][k]
                x, y = dic_of_shp['shape'][k].exterior.xy
                fig_c = fn_gen_plotly_scatter(fig_c, list(x), list(y),
                                              row=None, col=None,
                                              # color='white', opacity=0.8, size=4,
                                              text=vill, opacity=1,
                                              update_layout=False,
                                              line_color='white', line_width=1.5,
                                              mode='lines', colorscale=None)

            fn_dbg("fn_gen_web_eda 3-1-1-2")  # 1463 ms

            # if bc_vill.endswith('里'):
            #     k = f'{"臺北市"}, {bc_dist}, {bc_vill}'
            #     fig_c = fn_plot_shp(fig_c, dic_of_shp, k, text=k)

            fn_dbg("fn_gen_web_eda 3-1-1-3")

        # if build_case_sel != '不限':
        #     df_1_sel = df_1[df_1['建案名稱'] == build_case_sel]
        #     x, y = df_1_sel[['經度']].values[0], df_1_sel[['緯度']].values[0]
        #     ave_price = df_1_sel['coor_ave'].values[0]
        #     text = f'{build_case_sel} {ave_price}萬/坪'
        #     fig_c = fn_gen_plotly_scatter(fig_c, x, y,
        #                                   row=None, col=None,
        #                                   color='red', text=text, opacity=0.9, marker_sym=4, size=13,
        #                                   update_layout=False)

        if '顯示' in bc_show:
            for bc in df_1['建案名稱'].unique():
                if str(bc) != 'nan':
                    df_1_sel = df_1[df_1['建案名稱'] == bc]
                    x, y = df_1_sel[['經度']].values[0], df_1_sel[['緯度']].values[0]
                    ave_price = df_1_sel['coor_ave'].values[0]
                    text = f'{bc} {ave_price}萬/坪'
                    fig_c = fn_gen_plotly_scatter(fig_c, x, y,
                                                  row=None, col=None,
                                                  color='red', text=text, opacity=0.9, marker_sym=4, size=14,
                                                  update_layout=False)

        x_range = [min(df_1['經度']), max(df_1['經度'])]
        y_range = [min(df_1['緯度']), max(df_1['緯度'])]

        yr_min, yr_max = min(df_1['交易年']), max(df_1['交易年'])
        # title = f'民國 {yr_min}年 ~ {yr_max}年 {"台北市" if dist == "不限" else dist} 預售屋 成交單價 蛋黃圖' \
        #         f' {"" if build_case_sel == "不限" else f"(❌: {build_case_sel} )"}'
        title = f'民國 {yr_min}年 ~ {yr_max}年 {"台北市" if dist == "不限" else dist} 預售屋 成交單價 蛋黃圖'

        fig_c.update_layout(title={'text': title,
                                   'x': 0.5,
                                   'xanchor': 'center'},
                            autosize=False,
                            margin={'l': 50, 'r': 20, 't': 30, 'b': 20},
                            xaxis_range=x_range, yaxis_range=y_range, width=800, height=550)

        tab_egg, tab_3D = st.tabs(['蛋黃圖', '立體圖'])

        with tab_egg:
            st.plotly_chart(fig_c)

        with tab_3D:
            st.plotly_chart(fig_3d)

        fn_dbg('fn_gen_web_eda 3-1-2')

    with tab_dist_char:  # st.expander(f'👓 檢視 物件特徵 的 分布狀況'):
        df_1, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='ch',
                                                             sel_option=['dist'])

        fig_bar_1, fig_bar_2, fig_bar_3, fig_bar_4, fig_bar_5 = fn_gen_analysis_statistic(df_1)

        tb_trade, tb_build_1, tb_build_2, tb_build_3, tb_pk = st.tabs(
            ['交易特徵分布', '建物特徵分布1', '建物特徵分布2', '建物特徵分布3', '車位特徵分布'])

        dist_sel = df_1['鄉鎮市區'].unique()
        dist_sel = '台北市' if len(dist_sel) > 1 else dist_sel[0]
        dist_sel = dist_sel + ' 預售屋'

        with tb_trade:
            c1, c2, c3 = st.columns([1.2, 2, 1])
            c2.write(f'{dist_sel} 交易特徵分布')
            st.plotly_chart(fig_bar_1, config=config)

        with tb_build_1:
            c1, c2, c3 = st.columns([1.2, 2, 1])
            c2.write(f'{dist_sel} 建物特徵分布1')
            st.plotly_chart(fig_bar_2, config=config)

        with tb_build_2:
            c1, c2, c3 = st.columns([1.2, 2, 1])
            c2.write(f'{dist_sel} 建物特徵分布2')
            st.plotly_chart(fig_bar_5, config=config)

        with tb_build_3:
            c1, c2, c3 = st.columns([1.2, 2, 1])
            c2.write(f'{dist_sel} 建物特徵分布3')
            st.plotly_chart(fig_bar_4, config=config)

        with tb_pk:
            c1, c2, c3 = st.columns([1.2, 2, 1])
            c2.write(f'{dist_sel} 車位特徵分布')
            st.plotly_chart(fig_bar_3, config=config)

        fn_dbg('fn_gen_web_eda 3-1-3')

    with tab_ana_corr:  # st.expander(f'👓 檢視 每坪單價 與 "各項" 指標 的關係'):
        df_1, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='all',
                                                             sel_option=['dist'])

        df_1 = df_1[df_1['地下樓層'].apply(lambda x: str(x).isnumeric())]
        df_1['地下樓層'] = df_1['地下樓層'].astype(int)

        df_1 = df_1[df_1['容積率(%)'].apply(lambda x: str(x).isnumeric())]
        df_1['容積率(%)'] = df_1['容積率(%)'].astype(int)

        options = ['捷運', '小學', '建物', '均價', '所得1', '所得2', '建案1', '建案2', '歷年訴訟']
        # cmp = st.radio('比較指標:', options=options, index=6)
        # fn_set_radio_2_hor()

        tb_mrt, tb_sku, tb_build, tb_ave, tb_income1, tb_income2, tb_build_case1, tb_build_case2, tb_lg = st.tabs(
            options)

        df_lg = df_1[df_1['建商訴訟'] > 0]
        df_lg = df_lg[df_lg['營造訴訟'] > 0]

        # title = f'每坪單價 與 "{cmp}" 指標 的關係'
        target = [dict(label='每坪單價', values=df_1['每坪單價(萬)'])]

        dimensions = [
            dict(label='通勤時間', values=df_1['MRT_Commute_Time_UL']),
            dict(label='捷運距離', values=df_1['MRT_DIST']),
            dict(label='進站人數', values=df_1['MRT_Tput_UL']),
            dict(label='出站人數', values=df_1['MRT_Tput_DL']),

            dict(label='小學距離', values=df_1['sku_dist']),
            dict(label='小學人數', values=df_1['sku_109_total']),
            dict(label='經度', values=df_1['經度']),
            dict(label='緯度', values=df_1['緯度']),

            dict(label='交易年度', values=df_1['交易年']),
            dict(label='建物坪數', values=df_1['建物坪數']),
            dict(label='交易樓層', values=df_1['移轉層次']),
            dict(label='總樓層數', values=df_1['總樓層數']),

            dict(label='座標平均', values=df_1['coor_ave']),
            dict(label='學區平均', values=df_1['SKU_ave']),
            dict(label='捷運平均', values=df_1['MRT_ave']),
            dict(label='行政區平均', values=df_1['DIST_ave']),

            dict(label='各里所得_總額', values=df_1['稅_綜合所得總額']),
            dict(label='各里所得_平均數', values=df_1['稅_平均數']),
            dict(label='各里所得_中位數', values=df_1['稅_中位數']),
            dict(label='平均減中位', values=df_1['稅_平均_減_中位']),

            dict(label='各里所得_第一分位', values=df_1['稅_第一分位數']),
            dict(label='各里所得_第三分位', values=df_1['稅_第三分位數']),
            dict(label='各里所得_標準差', values=df_1['稅_標準差']),
            dict(label='各里所得_變異數', values=df_1['稅_變異係數']),

            dict(label='總戶數', values=df_1['總戶數']),
            dict(label='基地面積(坪)', values=df_1['基地面積(坪)']),
            dict(label='地下樓層', values=df_1['地下樓層']),
            dict(label='完工年度', values=df_1['完工年度']),

            dict(label='建蔽率(%)', values=df_1['建蔽率(%)']),
            dict(label='容積率(%)', values=df_1['容積率(%)']),
            dict(label='公設比(%)', values=df_1['公設比(%)']),
            # dict(label='公設比(%)', values=df_1['公設比(%)']),

            dict(label='建商 歷年訴訟(件)', values=df_lg['建商訴訟']),
            dict(label='營造 歷年訴訟(件)', values=df_lg['營造訴訟']),
            dict(label='建商+營造 歷年訴訟(件)', values=df_lg['所有訴訟']),
            # dict(label='公設比(%)', values=df_1['公設比(%)']),
        ]

        figs = 4
        d1 = dimensions[:figs]
        d2 = dimensions[figs: 2 * figs]
        d3 = dimensions[2 * figs: 3 * figs]
        d4 = dimensions[3 * figs: 4 * figs]
        d5 = dimensions[4 * figs: 5 * figs]
        d6 = dimensions[5 * figs: 6 * figs]
        d7 = dimensions[6 * figs: 7 * figs]
        d8 = dimensions[7 * figs: 8 * figs - 1]
        d9 = dimensions[8 * figs - 1: 9 * figs - 1]

        plots = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
        dic_of_show = {k: plots[options.index(k)] for k in options}

        with tb_mrt:
            cmp = options[0]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_sku:
            cmp = options[1]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_build:
            cmp = options[2]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_ave:
            cmp = options[3]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_income1:
            cmp = options[4]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_income2:
            cmp = options[5]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_build_case1:
            cmp = options[6]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_build_case2:
            cmp = options[7]
            fn_corr_util(dic_of_show, df_1, cmp, target, config)

        with tb_lg:
            cmp = options[8]
            fn_corr_util(dic_of_show, df_lg, cmp, target, config)

        fn_dbg('fn_gen_web_eda 3-1-4')

    with tab_ana_dist:  # st.expander(f'👓 檢視 每坪單價 與 "行政區" 指標 的關係'):
        # color_by = st.radio('著色條件:', options=['無', f'依最新登錄({latest_records})'], index=0)
        # fn_set_radio_2_hor()
        figs = fn_gen_analysis_admin(df, bc_name=[build_case])
        st.plotly_chart(figs[0], config=config)
        if len(figs) > 1:
            st.plotly_chart(figs[1], config=config)
        fn_dbg('fn_gen_web_eda 3-1-5')

    with tab_ana_mrt:  # st.expander(f'👓 檢視 每坪單價 與 "捷運" 指標 的關係'):
        colors = ['無', '依捷運距離', '依通勤時間', f'依最新登錄({latest_records})']
        # color_by = st.radio('著色條件:', options=colors, index=0)
        # fn_set_radio_2_hor()

        df_sel, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='mrt',
                                                               colors=colors)
        fig_sct, fig_sct_1 = fn_gen_analysis_mrt(df_sel, color_by, bc_name=[build_case_sel])
        st.plotly_chart(fig_sct, config=config)
        st.plotly_chart(fig_sct_1, config=config)
        fn_dbg('fn_gen_web_eda 3-1-6')

    with tab_ana_ele:  # st.expander(f'👓 檢視 每坪單價 與 "小學" 指標 的關係'):
        colors = ['無', '依小學距離', '依小學人數', f'依最新登錄({latest_records})']
        # color_by = st.radio('著色條件:', options=colors, index=0)
        # fn_set_radio_2_hor()
        df_sel, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='sku',
                                                               colors=colors)

        fig_sku_1, fig_sku_2 = fn_gen_analysis_sku(df_sel, color_by, bc_name=[build_case_sel])
        st.plotly_chart(fig_sku_1, config=config)
        st.plotly_chart(fig_sku_2, config=config)
        fn_dbg('fn_gen_web_eda 3-1-7')

    with tab_ana_bd:  # st.expander(f'👓 檢視 每坪單價 與 "建物" 指標 的關係'):
        df_sel, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records)
        r = st.radio('價格選項', ['每坪單價(萬)', '總價(萬)'], index=0)
        if r == '每坪單價(萬)':
            fig_sct_3 = fn_gen_analysis_building(df_sel, '每坪單價(萬)', color_by, bc_name=[build_case_sel])
            st.plotly_chart(fig_sct_3, config=config)
        elif r == '總價(萬)':
            fig_sct_3 = fn_gen_analysis_building(df_sel, '總價(萬)', color_by, bc_name=[build_case_sel])
            st.plotly_chart(fig_sct_3, config=config)

        fn_dbg('fn_gen_web_eda 3-1-8')

    with tab_ana_sell:  # st.expander(f'👓 檢視 "銷售分析"'):
        df_sel, build_case_sel, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='period')
        fig_gantt, fig_bar = fn_gen_analysis_sale_period(df_sel, build_case_sel)
        st.plotly_chart(fig_gantt, config=config)
        st.write('')
        st.plotly_chart(fig_bar, config=config)
        fn_dbg('fn_gen_web_eda 3-1-9')

    # fn_dbg('fn_gen_web_eda 3-1-10')


@fn_profiler
def fn_gen_bc_deals_old(build_case, dic_df_show):
    if len(dic_df_show.keys()):
        deals = np.count_nonzero(dic_df_show['每坪單價(萬)'])
        st.write('')
        st.subheader(f'🏡 建案: {build_case}'
                     f' 📝 登錄: {deals} 筆'
                     f' 💰 總金額: {round((dic_df_show["總價(萬)"].values.sum()) / 10000, 2)} 億')

        r = st.radio('檢視選項:', options=['每坪單價(萬)', '樓層價差(%)', '總價-車位(萬)', '總價(萬)', '車位總價(萬)', '建物坪數', '車位坪數', '交易日期'],
                     index=0)
        fn_set_radio_2_hor()

        dic_df_show['樓層價差(%)'] = dic_df_show['每坪單價(萬)']

        df_show = dic_df_show[r] if r in dic_df_show.keys() else None

        df_show = df_show[df_show.index != '1F']

        if r == '樓層價差(%)':
            df_show_diff = df_show.copy()
            rows, cols = df_show_diff.shape[0], df_show_diff.shape[1]
            for idx in range(rows - 1):
                for col in range(cols):
                    son = df_show.iloc[idx, col]
                    mom = df_show.iloc[idx + 1, col]
                    f = df_show.index[idx]
                    f_1 = df_show.index[idx + 1]
                    is_f_cont = abs(int(f.split('F')[0]) - int(f_1.split('F')[0])) == 1
                    if is_f_cont and son > 0 and mom > 0:
                        df_show_diff.at[f, df_show.columns[col]] = round(son / mom, 4) - 1

            for col in df_show_diff:
                df_show_diff[col] = df_show_diff[col].apply(lambda x: 0 if x > 2 else x)

            df_show = df_show_diff

        assert df_show is not None, f'{r} not in dic_df_show {dic_df_show.keys()}'

        if r in ['每坪單價(萬)', '建物坪數', '車位坪數']:
            fmt = "{:.2f}"
        elif r in ['樓層價差(%)']:
            fmt = "{:.1%}"
        else:
            fmt = None

        df_show = df_show.astype(int) if r == '交易日期' else df_show
        df_show_fig = df_show.style.format(fmt).applymap(fn_gen_df_color)

        sorts = []
        for col in df_show.columns:
            sorts += list(df_show[col].values)

        sorts = [v for v in sorts if v > 0]
        sorts.sort()

        df_show_fig = df_show_fig.background_gradient(cmap='rainbow', low=0.8, high=0, axis=None, vmin=sorts[0])
        df_show_fig = df_show_fig.highlight_between(left=0, right=0.0005, axis=1, color='gray')

        st.dataframe(df_show_fig, width=768, height=540)
        dic_values = defaultdict(list)
        for col in df_show.columns:
            for idx in df_show.index:
                v = df_show.loc[idx, col]
                a = int(dic_df_show['建物坪數'].loc[idx, col])
                if v > 0:
                    if r == '交易日期':
                        year = int(v / 100)
                        month = v - 100 * year
                        v = datetime.date(year=year, month=month, day=1)
                    dic_values[a].append(v)

        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=(
                                f'建案-{build_case}: {len(dic_values.keys())}種坪數 共{deals}筆交易 的 "{r}" 分布',))

        dic_values_sort = {k: dic_values[k] for k in sorted(dic_values)}

        margin = {'l': 40}
        for k in dic_values_sort.keys():
            fig = fn_gen_plotly_hist(fig, dic_values_sort[k], f'{str(k)}坪{r}', bins=50, margin=margin,
                                     line_color='black', showlegend=True)

        with st.expander('銷售分析'):
            st.plotly_chart(fig)


@fn_profiler
def fn_gen_bc_deals(build_case, dic_df_show, r):
    fn_gen_bc_summary(dic_df_show, r)
    if len(dic_df_show.keys()):

        dic_df_show['樓層價差(%)'] = dic_df_show['每坪單價(萬)']

        if r == '單價分布':
            df_show = dic_df_show['每坪單價(萬)']
        else:
            df_show = dic_df_show[r] if r in dic_df_show.keys() else None

        df_show = df_show[df_show.index != '1F']

        # st.write(df_show)  # test

        if r == '樓層價差(%)':
            df_show_diff = df_show.copy()
            rows, cols = df_show_diff.shape[0], df_show_diff.shape[1]
            for idx in range(rows - 1):
                for col in range(cols):
                    son = df_show.iloc[idx, col]
                    mom = df_show.iloc[idx + 1, col]
                    f = df_show.index[idx]
                    f_1 = df_show.index[idx + 1]
                    is_f_cont = abs(int(f.split('F')[0]) - int(f_1.split('F')[0])) == 1
                    if is_f_cont and son > 0 and mom > 0:
                        df_show_diff.at[f, df_show.columns[col]] = round(son / mom, 4) - 1

            for col in df_show_diff:
                df_show_diff[col] = df_show_diff[col].apply(lambda x: 0 if x > 2 else x)

            df_show = df_show_diff

        assert df_show is not None, f'{r} not in dic_df_show {dic_df_show.keys()}'

        if r in ['每坪單價(萬)', '建物坪數', '車位坪數']:
            fmt = "{:.2f}"
        elif r in ['樓層價差(%)']:
            fmt = "{:.1%}"
        else:
            fmt = None

        df_show = df_show.astype(int) if r == '交易日期' else df_show

        df_show_fig = df_show.style.format(fmt).applymap(fn_gen_df_color)

        if r == '單價分布':
            pass
        else:
            sorts = []
            for col in df_show.columns:
                sorts += list(df_show[col].values)

            if r == '樓層價差(%)':
                vmin = min(sorts)
            else:
                sorts = [v for v in sorts if v > 0]
                sorts.sort()
                vmin = sorts[0] if len(sorts) > 0 else 0

            df_show_fig = df_show_fig.background_gradient(cmap='rainbow', low=0.8, high=0, axis=None, vmin=vmin)
            df_show_fig = df_show_fig.highlight_between(left=0, right=0.0005, axis=1, color='gray')

            st.dataframe(df_show_fig, width=768, height=540)

        dic_values = defaultdict(list)
        for col in df_show.columns:
            for idx in df_show.index:
                v = df_show.loc[idx, col]
                a = int(dic_df_show['建物坪數'].loc[idx, col])
                if v > 0:
                    if r == '交易日期':
                        year = int(v / 100)
                        month = v - 100 * year
                        v = datetime.date(year=year, month=month, day=1)
                    dic_values[a].append(v)

        # if r == '每坪單價(萬)':
        if r == '單價分布':
            deals = np.count_nonzero(dic_df_show['每坪單價(萬)'])
            fig = make_subplots(rows=1, cols=1,
                                subplot_titles=(
                                    f'建案-{build_case}: {len(dic_values.keys())}種坪數 共{deals}筆交易 的 "{r}" 分布',))

            dic_values_sort = {k: dic_values[k] for k in sorted(dic_values)}

            margin = {'l': 40, 't': 30}
            colors = plotly.colors.qualitative.Vivid + plotly.colors.qualitative.Set3 + plotly.colors.qualitative.Light24
            c = 0
            assert len(colors) > len(dic_values_sort.keys()), f'colors {len(colors)} > c {c} Fail !'
            for k in dic_values_sort.keys():
                fig = fn_gen_plotly_hist(fig, dic_values_sort[k], f'{str(k)}坪{r}', bins=50, margin=margin,
                                         line_color='black', color=colors[c], showlegend=True)
                c = c + 1

            # with st.expander('銷售分析'):
            st.plotly_chart(fig)


@fn_profiler
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


def fn_dbg(msg='', is_force=False):
    if dic_of_dbg['is_dbg'] or is_force:
        now = int(round(time.time() * 1000, 0))  # ms
        dur = 0 if dic_of_dbg['time'] == 0 else now - dic_of_dbg['time']
        dic_of_dbg['time'] = now
        dic_of_dbg['sum'] = dic_of_dbg['sum'] + dur
        note = '!!!' if dur > 1000 else ''
        if msg == '':
            pass
        else:
            if is_force:
                st.sidebar.success(f'{dur} / {dic_of_dbg["sum"]} ms --> {msg} {note}')
                st.sidebar.write('')
            else:
                st.sidebar.write(f'{dur} / {dic_of_dbg["sum"]} ms --> {msg} {note}')


def fn_util_split(constructor):
    # sep = ['、', ',', '-', '/', 'X', '(']
    if '/' in constructor:
        constructors = constructor.split('/')
    elif ',' in constructor:
        constructors = constructor.split(',')
    elif '、' in constructor:
        constructors = constructor.split('、')
    elif '-' in constructor:
        constructors = constructor.split('-')
    elif 'X' in constructor:
        constructors = constructor.split('X')
    else:
        constructors = [constructor]

    return constructors


def fn_gen_bc_summary(dic_df_show, key):
    dic_summary = {
        '每坪單價(萬)': ['最高單價', '最低單價', '(萬/坪)'],
        '總價(萬)': ['最高總價', '最低總價', '(萬)'],
        '總價-車位(萬)': ['最高總價', '最低總價', '(萬)'],
        '車位總價(萬)': ['最高總價', '最低總價', '(萬)'],
        # '建物坪數': ['最大坪數', '最小坪數', '(坪)'],
        # '車位坪數': ['最大坪數', '最小坪數', '(坪)'],
    }

    if key in dic_summary.keys():
        df = dic_df_show[key]
        v_max, v_min = df.max().max(), df[df > 0].min().min()
        dic_max, dic_min = {}, {}
        for c in df.columns:
            if v_max in df[c].values:
                dic_max[c] = df[df[c] == v_max].index.values.tolist()

            if v_min in df[c].values:
                dic_min[c] = df[df[c] == v_min].index.values.tolist()

        str_max, str_min = ' ', ' '
        for k, v in dic_max.items():
            str_max = str_max + f'{k}-{f", {k}-".join(v)}, '

        for k, v in dic_min.items():
            str_min = str_min + f'{k}-{f", {k}-".join(v)}, '

        v_min = int(v_min) if '總' in key else v_min
        st.write(f'{dic_summary[key][0]} 👉 {v_max} {dic_summary[key][2]} : {str_max[:-2]}')
        st.write(f'{dic_summary[key][1]} 👉 {v_min} {dic_summary[key][2]} : {str_min[:-2]}')
    else:
        st.write('')
        st.write('')
        st.write('')
        st.write('')


def fn_create_chart(df):
    df['交易年'] = df['交易年'].astype(str) + '年'
    df['交易筆數'] = df['count']
    # initialize chart
    chart = Chart(width="320px", height="680px", display=DisplayTarget.MANUAL)
    # add data
    data = Data()
    data.add_data_frame(df)
    chart.animate(data)

    chart.animate(Config({"x": "交易筆數", "y": "鄉鎮市區", "label": "count", "title": "台北市預售屋"}))
    chart.animate(Config({"x": "交易筆數", "y": ["鄉鎮市區", "交易年"], "label": ["count", "交易年"], "color": "交易年"}))

    # add style
    chart.animate(Style({"title": {"fontSize": 24}}))

    return chart._repr_html_()


def fn_create_slide(df):
    df['交易筆數'] = df['count']
    years = df['交易年'].unique()
    df['交易年'] = df['交易年'].astype(str) + '年'
    data = Data()

    data.add_data_frame(df)
    story = Story(data=data)
    title = f"{min(years)}年 ~ {max(years)}年 台北市預售屋銷售"
    slide1 = Slide(
        Step(
            Config({"x": "交易筆數", "y": ["鄉鎮市區", "交易年"], "label": ["count", "交易年"], "color": "交易年", "title": f"{title}"}),
            Style({"title": {"fontSize": 24}}),
        )
    )
    story.add_slide(slide1)

    slide2 = Slide(
        Step(
            Config({"x": "交易筆數", "y": "鄉鎮市區", "label": "count", "color": None}),
            Style({"title": {"fontSize": 24}}),
        )
    )
    story.add_slide(slide2)

    story.set_size(width=460, height=680)

    story.play()


# =========

def fn_gen_bc2show(df_sel):
    dic_df_show = dict()
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

    df_bc.fillna(round(0, 1), inplace=True)
    df_bc_t.fillna(round(0, 1), inplace=True)
    df_bc_car.fillna(round(0, 1), inplace=True)
    df_bc_s.fillna(round(0, 1), inplace=True)
    df_bc_ps.fillna(round(0, 1), inplace=True)
    df_bc_d.fillna(round(0, 1), inplace=True)

    # if floor != 0:
    #     df_bc = df_bc[df_bc.index == str(floor) + 'F']
    #     df_bc_t = df_bc_t[df_bc_t.index == str(floor) + 'F']
    #     df_bc_car = df_bc_car[df_bc_car.index == str(floor) + 'F']
    #     df_bc_s = df_bc_s[df_bc_s.index == str(floor) + 'F']
    #     df_bc_ps = df_bc_ps[df_bc_ps.index == str(floor) + 'F']
    #     df_bc_d = df_bc_d[df_bc_d.index == str(floor) + 'F']

    dic_df_show['每坪單價(萬)'] = df_bc[df_bc.sum(axis=1) > 0]
    dic_df_show['總價(萬)'] = df_bc_t[df_bc_t.sum(axis=1) > 0]
    dic_df_show['車位總價(萬)'] = df_bc_car[df_bc_car.sum(axis=1) > 0]
    dic_df_show['建物坪數'] = df_bc_s[df_bc_s.sum(axis=1) > 0]
    dic_df_show['車位坪數'] = df_bc_ps[df_bc_ps.sum(axis=1) > 0]
    # dic_df_show['建物-車位(坪)'] = dic_df_show['建物坪數'] - dic_df_show['車位坪數']
    dic_df_show['總價-車位(萬)'] = dic_df_show['總價(萬)'] - dic_df_show['車位總價(萬)']
    dic_df_show['交易日期'] = df_bc_d[df_bc_d.sum(axis=1) > 0] / 100
    # print(f'{dic_df_show["建物坪數"] }')

    return dic_df_show


def fn_gen_floor_ave(df_sel):
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
    title = '樓層均價'  # f'{mrt} ({From_To})'
    fig_bar2 = go.Figure(data=[
        go.Bar(name='均價(萬/坪)', x=floors, y=prices, opacity=0.7),
        go.Bar(name='成交戶數', x=floors, y=deals, opacity=0.7)
    ],
        layout={'title': title},
        layout_yaxis_range=[min(prices)-5, max(prices)+5])

    fig_bar2.update_layout(barmode='group',  # One of 'group', 'overlay' or 'relative'
                           margin=dict(l=30, r=20, t=60, b=40),
                           # paper_bgcolor="LightsteelBlue",
                           font=dict(size=16))

    return fig_bar2


@fn_profiler
def fn_gen_web_eda(df):
    # t_s = time.time()

    df_tm = df[['台北市', '鄉鎮市區', '每坪單價(萬)', '建案名稱', '建物坪數']]
    df_tm = df_tm[df_tm['台北市'] == 1]
    df_tm = df_tm[df_tm['建案名稱'].apply(lambda x: str(x) != 'nan')]
    df_tm_v = pd.DataFrame(df_tm.groupby('建案名稱', as_index=True)['每坪單價(萬)'].mean())
    df_tm_s = pd.DataFrame(df_tm.groupby('建案名稱', as_index=True)['建物坪數'].mean())
    df_tm_m = pd.DataFrame(df_tm.groupby('建案名稱', as_index=True)['建物坪數'].max())
    df_tm_m.rename(columns={'建物坪數': '最大坪數'}, inplace=True)
    df_tm_n = pd.DataFrame(df_tm.groupby('建案名稱', as_index=True)['建物坪數'].min())
    df_tm_n.rename(columns={'建物坪數': '最小坪數'}, inplace=True)
    df_tm_c = pd.DataFrame(df_tm.groupby('建案名稱', as_index=True)['建案名稱'].count())
    df_tm_v = df_tm_v['每坪單價(萬)'].apply(lambda x: round(x, 2))
    df_tm_s = df_tm_s['建物坪數'].apply(lambda x: round(x, 2))
    df_tm = pd.concat([df_tm_v, df_tm_c], axis=1)
    df_tm = pd.concat([df_tm, df_tm_s, df_tm_m, df_tm_n], axis=1)
    df_tm.sort_values(by='每坪單價(萬)', inplace=True)

    for i in df_tm.index:
        df_d = df[df['建案名稱'] == i]
        df_tm.at[i, '城市'] = '台北市'
        df_tm.at[i, '行政區'] = df_d['鄉鎮市區'].values[0]
        df_tm.at[i, '捷運'] = df_d['MRT'].values[0]
        df_tm.at[i, '小學'] = df_d['sku_name'].values[0]
        df_tm.at[i, '交易年'] = df_d['交易年'].values[0]

    df_tm.rename(columns={'建案名稱': '交易筆數'}, inplace=True)
    df_tm.reset_index(inplace=True)
    df_tm.rename(columns={'index': '建案名稱', '每坪單價(萬)': '每坪均價(萬)'}, inplace=True)

    df_sel = df.copy()
    # df_sel = df_sel.reset_index(drop=True)

    Latest_date = str(df_sel['交易年月日'].iloc[0])
    Latest_date = Latest_date[0:-4] + '年' + Latest_date[-4].replace('0', '') + Latest_date[-3] + '月'

    options = list(df_sel[['MRT']].sort_values(by='MRT')['MRT'].unique()) + ['不限']
    idx = options.index('不限') if '不限' in options else 0
    mrt = st.sidebar.selectbox('捷運站', options=options, index=idx)
    df_sel = df_sel.reset_index(drop=True) if mrt == '不限' else df_sel[df_sel['MRT'] == mrt].reset_index(drop=True)

    df_latest = df_sel[df_sel['交易年月日'] == max(df_sel['交易年月日'])]
    build_case = df_latest['建案名稱'].values[0]
    df_sel = df_sel[df_sel['建案名稱'] == build_case].reset_index(drop=True) if build_case != '不限' else df_sel

    for i in range(5):
        st.sidebar.write('')

    fn_dbg('fn_gen_web_eda 1')

    From = str(df_sel['交易年月日'].iloc[-1])
    From = From[0:-4] + '年' + From[-4].replace('0', '') + From[-3] + '月'
    To = str(df_sel['交易年月日'].iloc[0])
    To = To[0:-4] + '年' + To[-4].replace('0', '') + To[-3] + '月'

    # From_To = f'{From} ~ {To}, 有 {len(df_sel)} 筆交易'

    deal_num = 0
    for b in df_sel['建案名稱'].unique():
        df_sel_b = df_sel[df_sel['建案名稱'] == b]
        deal_num = deal_num + len(df_sel_b["戶別"].unique())

    deal_num = df_sel.shape[0] if build_case == '不限' else deal_num
    From_To = f'{From} ~ {To}, 有 {deal_num} 筆交易'
    ave = round(df_sel['每坪單價(萬)'].mean(), 0)

    # df_bc = pd.DataFrame()
    # dic_df_show = dict()

    df_sel.rename(columns={'log': 'lon'}, inplace=True)  # rename for st.map

    # df_sel['每坪單價(萬)']=df_sel['每坪單價(萬)'].astype(int)
    df_sel['MRT_DIST'] = df_sel['MRT_DIST'].astype(int)

    df_sel.rename(columns={'MRT': '捷運站', 'MRT_DIST': '捷運站距離(m)'}, inplace=True)

    house_typ = '預售屋' if len(df['建築完成年月'].unique()) == 1 else '中古屋'
    city = df['土地位置建物門牌'].apply(lambda x: str(x).split('市')[0] + '市')
    city = city.unique()[0]


    period = f"$民國 {df['交易年'].min()} \sim {df['交易年'].max()}年$"
    b_typ = '' if df["建物型態"].nunique() > 1 else df["建物型態"].values[0] + '型 '
    title = f'{period} :blue[${city}$] ${df["建案名稱"].nunique()}個$ :blue[${b_typ}\ {house_typ}$] $建案 共$ :red[${df.shape[0]}筆$] $實價登錄資料$'

    map_style = "carto-positron"  # "open-street-map"
    df = df.sort_values(by=['交易年月日'])

    df['地址_建案'] = df['地址'] + df['建案名稱']

    col_grp = '地址_建案'

    df_bc_1 = pd.DataFrame(df.groupby(col_grp, as_index=True)['地址'].count()).rename(columns={'地址': '交易量'})
    df_bc_2 = pd.DataFrame(df.groupby(col_grp, as_index=True)['MRT'].nth(1))
    df_bc_3 = pd.DataFrame(df.groupby(col_grp, as_index=True)['建案名稱'].nth(1))
    df_bc_4 = pd.DataFrame(df.groupby(col_grp, as_index=True)['交易年月日'].nth(-1)).rename(columns={'交易年月日': '最新登錄'})
    df_bc_5 = pd.DataFrame(df.groupby(col_grp, as_index=True)['經度'].nth(1))
    df_bc_6 = pd.DataFrame(df.groupby(col_grp, as_index=True)['緯度'].nth(1))
    df_bc_7 = pd.DataFrame(df.groupby(col_grp, as_index=True)['每坪單價(萬)'].mean()).rename(columns={'每坪單價(萬)': '每坪均價(萬)'})

    df_bc_cnt = pd.concat([df_bc_1, df_bc_2, df_bc_3, df_bc_4, df_bc_5, df_bc_6, df_bc_7], axis=1)
    df_bc_cnt['每坪均價(萬)'] = df_bc_cnt['每坪均價(萬)'].apply(lambda x: round(x, 2))

    hover_name = "建案名稱"
    hover_data = ["MRT", '最新登錄']
    color = '每坪均價(萬)'

    fig_map_all = fn_gen_plotly_map(df_bc_cnt, None, hover_name, hover_data, map_style, color=color, zoom=10.5,
                                    op=0.55,
                                    size='交易量')

    latest_rel = '0921'
    records = int(df.shape[0] - np.count_nonzero(df['Latest']))
    latest_records = f'版本:{latest_rel} 有 {records}筆'
    city = list(df['city'].unique())
    cities = ''
    for c in city:
        cities = cities + c + ' '

    dist = '' if df['鄉鎮市區'].nunique() > 1 else df['鄉鎮市區'].unique()[0]

    # rendering web view
    st.write('')

    # st.header(f'🏙️ {cities}{dist} {house_typ} 實價登錄 (最新:{Latest_date}) ')
    c1, c2 = st.columns([4, 1.1])
    c1.header(f'🏙️ {cities}{dist} {house_typ} 實價登錄 ')
    # c1.markdown(f'### 🏙️ $ {cities}{dist}\ {house_typ}\ 實價登錄 $')
    # c1.markdown(f'## 🏙️ $ {cities}{dist}\ {house_typ}\ 實價登錄 $')

    latest_file = df["File"].values[-1]
    latest_records = df[df["File"] == latest_file].shape[0]
    pre_file = df[df["File"] != latest_file]["File"].values[-1]

    pre_records = df[df["File"] == pre_file].shape[0]
    delta = latest_records - pre_records
    # st.write(f'{latest_file} {latest_records} {pre_file} {pre_records} {delta}')
    if latest_file.split('.')[0].endswith('p'):
        latest_date = str(latest_file).lower().split('_b_')[-1].split('.')[0].split('_')[-2]
    else:
        latest_date = str(latest_file).lower().split('_b_')[-1].split('.')[0].split('_')[-1]
    latest_date = latest_date[:2] + '/' + latest_date[2:] if latest_date.isnumeric() else latest_date
    pre_date = str(pre_file).lower().split('_b_')[-1].split('.')[0].split('_')[-1]
    pre_date = pre_date[:2] + '/' + pre_date[2:] if pre_date.isnumeric() else pre_date
    c2.metric(f'本期更新: {latest_date} 🌟', f'{latest_records} 筆', f'{delta} 筆(前期: {pre_date})', delta_color='inverse')
    # st.write(df.head())

    # tabs = st.tabs([f'{cities}實價登錄', '台北市均價', '行政區均價', '交易筆數', '最小坪數', '最大坪數', '價格走勢', '交易量走勢'])
    # tab_price_map, tab_price_tpe, tab_price, tab_deals, tab_area_min, tab_area_max, tab_trend_price, tab_trend_amount = tabs

    tabs = st.tabs(['預售總覽', f'預售地圖', '行政區均價', '價格走勢', '交易量走勢'])
    tab_overview, tab_price_map, tab_price, tab_trend_price, tab_trend_amount = tabs

    with tab_overview:
        st.write(f'{title}')
        df_s = df[['交易年月日', '鄉鎮市區', '每坪單價(萬)', '交易年']]
        df_s['count'] = 1

        # story = fn_create_chart(df_s.copy())
        # html(story, width=800, height=700)

        fn_create_slide(df_s.copy())

    with tab_trend_price:
        st.write(f'{title}')
        df_plost = df[['交易年月日', '鄉鎮市區', '每坪單價(萬)']]
        df_plost.reset_index(drop=True, inplace=True)
        df_plost['交易年月日'] = df_plost['交易年月日'].apply(lambda x: str(x + 19110000))
        df_plost['date'] = pd.to_datetime(df_plost['交易年月日'])
        df_plost['每坪單價(萬)'] = df_plost['每坪單價(萬)'].apply(lambda x: round(x, 2))
        df_plost.rename(columns={'每坪單價(萬)': '單價(萬)'}, inplace=True)

        df_plost['year'] = pd.DatetimeIndex(df_plost['date']).year
        df_plost['month'] = pd.DatetimeIndex(df_plost['date']).month
        df_yp = pd.DataFrame()
        for y in df_plost['year'].unique():
            df_y = df_plost[df_plost['year'] == y]
            df_y[f'單價(萬)_{y}'] = df_y[f'單價(萬)']
            df_ym = df_y.groupby('month', as_index=True)[f'單價(萬)_{y}'].mean()
            df_yp = pd.concat([df_yp, df_ym], axis=1)
        df_yp.reset_index(inplace=True)
        df_yp = df_yp.sort_values(by='index', ignore_index=True)
        df_yp['Month'] = pd.to_datetime(df_yp['index'], format="%m")

        this_y = datetime.datetime.today().year  # temp workaround, remove it later !
        last_y = this_y - 1
        v = 'NA'
        m = 'NA'
        d = 'NA'
        col_y = f'單價(萬)_{this_y}'
        col_yp = f'單價(萬)_{last_y}'
        if col_y in df_yp.columns:
            for idx in df_yp.index:
                if str(df_yp.loc[idx, col_y]) == 'nan':
                    try:
                        v = int(round(df_yp.loc[idx - 1, col_y], 0))
                        m = idx
                        vp = df_yp.loc[idx - 1, col_yp]
                        if str(vp) != 'nan':
                            d = int(round(v - df_yp.loc[idx - 1, col_yp], 0))
                    except:
                        # pass
                        v = int(round(df_yp.loc[idx - 1, col_y], 0))
                        m = idx
                        vp = df_yp.loc[idx - 1, col_yp]
                        if str(vp) != 'nan':
                            d = int(round(v - df_yp.loc[idx - 1, col_yp], 0))
                    break

        st.write('')
        place = city[-1] if dist == '' else dist
        c1, c2 = st.columns(2)

        c1.metric(f'{place} {this_y}年{m}月 均價', f'{v} 萬/坪', f'{d} 萬/坪(比較去年同期)', delta_color='inverse')

        c1, c2 = st.columns([1, 2])

        with c1:
            # st.markdown('### 均價')
            plost.time_hist(
                data=df_plost,
                date='date',
                x_unit='year',
                y_unit='month',
                color='單價(萬)',
                aggregate='average',
                legend=None,
                height=345,
                width=850,
                use_container_width=True)

        with c2:
            y = [c for c in df_yp.columns if '單價' in c]
            st.line_chart(df_yp, x='Month', y=y, height=345)

    with tab_trend_amount:
        st.write(f'{title}')

        df_yp = pd.DataFrame()
        for y in df_plost['year'].unique():
            df_y = df_plost[df_plost['year'] == y]
            df_y[f'單價(萬)_{y}'] = df_y[f'單價(萬)']
            df_ym = df_y.groupby('month', as_index=True)[f'單價(萬)_{y}'].count()
            df_yp = pd.concat([df_yp, df_ym], axis=1)
        df_yp.reset_index(inplace=True)
        df_yp['Month'] = pd.to_datetime(df_yp['index'], format="%m")
        df_yp.rename(columns={c: c.replace('單價(萬)', '交易量') for c in df_yp.columns if '單價(萬)' in c}, inplace=True)

        v = 'NA'
        m = 'NA'
        d = 'NA'
        col_y = f'交易量_{this_y}'
        col_yp = f'交易量_{this_y - 1}'
        if col_y in df_yp.columns:
            for idx in df_yp.index:
                if str(df_yp.loc[idx, col_y]) == 'nan':
                    try:
                        v = int(round(df_yp.loc[idx - 1, col_y], 0))
                        m = idx
                        vp = df_yp.loc[idx - 1, col_yp]
                        if str(vp) != 'nan':
                            d = int(round(v - df_yp.loc[idx - 1, col_yp], 0))
                    except:
                        pass
                        # st.write(idx)
                        # st.write(df_yp)
                    break

        st.write('')
        st.metric(f'{place} {this_y}年{m}月 交易量', f'{v} 筆', f'{d} 筆(比較去年同期)', delta_color='inverse')
        c1, c2 = st.columns([1, 2])
        with c1:
            plost.time_hist(
                data=df_plost,
                date='date',
                x_unit='year',
                y_unit='month',
                color='單價(萬)',
                aggregate='count',
                legend=None,
                height=345,
                width=820,
                use_container_width=True)

        with c2:
            y = [c for c in df_yp.columns if '交易量' in c]
            st.line_chart(df_yp, x='Month', y=y, height=345)

    with tab_price_map:
        st.write(f'{title}')
        st.plotly_chart(fig_map_all)
        st.write('')
        fn_dbg('fn_gen_web_eda 2')

    # with tab_price_tpe:
    #     fig_tm = fn_gen_plotly_treemap(df_tm, path=['城市', '建案名稱'], values='每坪均價(萬)',
    #                                    color='每坪均價(萬)', hover=['交易年', '捷運', '小學'],
    #                                    mid=np.average(df_tm['每坪均價(萬)'], weights=df_tm['交易筆數']))
    #     st.plotly_chart(fig_tm)
    #     fn_dbg('fn_gen_web_eda 2-1')

    with tab_price:
        st.write(f'{title}')
        fig_tm = fn_gen_plotly_treemap(df_tm, path=['城市', '行政區', '建案名稱'], values='每坪均價(萬)',
                                       color='每坪均價(萬)', hover=['交易年', '捷運', '小學'],
                                       mid=np.average(df_tm['每坪均價(萬)'], weights=df_tm['交易筆數']))
        st.plotly_chart(fig_tm)
        fn_dbg('fn_gen_web_eda 2-2')

    # with tab_deals:
    #     fig_tm = fn_gen_plotly_treemap(df_tm, path=['城市', '行政區', '建案名稱'], values='交易筆數',
    #                                    color='每坪均價(萬)', hover=['交易年', '捷運', '小學'],
    #                                    mid=np.average(df_tm['每坪均價(萬)'], weights=df_tm['交易筆數']))
    #     st.plotly_chart(fig_tm)
    #     fn_dbg('fn_gen_web_eda 2-3')
    #
    # with tab_area_min:
    #     fig_tm_n = fn_gen_plotly_treemap(df_tm, path=['城市', '行政區', '建案名稱'], values='最小坪數',
    #                                      color='每坪均價(萬)', hover=['交易年', '捷運', '小學'],
    #                                      mid=np.average(df_tm['每坪均價(萬)'], weights=df_tm['交易筆數']))
    #     st.plotly_chart(fig_tm_n)
    #     fn_dbg('fn_gen_web_eda 2-4')
    #
    # with tab_area_max:
    #     fig_tm_m = fn_gen_plotly_treemap(df_tm, path=['城市', '行政區', '建案名稱'], values='最大坪數',
    #                                      color='每坪均價(萬)', hover=['交易年', '捷運', '小學'],
    #                                      mid=np.average(df_tm['每坪均價(萬)'], weights=df_tm['交易筆數']))
    #     st.plotly_chart(fig_tm_m)
    #     fn_dbg('fn_gen_web_eda 2-5')

    fn_dbg('fn_gen_web_eda 3')

    st.write('')
    st.header(f'📊 數據分析')
    fn_gen_analysis(df.copy(), latest_records, build_case)

    st.write('')
    period = 12 * (int(To.split('年')[0]) - int(From.split('年')[0])) + \
             int(To.split('年')[-1].split('月')[0]) - int(From.split('年')[-1].split('月')[0]) + 1

    st.write('')

    # st.header(f'🏘️ 建案分析 👉 {build_case}')

    st.header(f'🏘️ 建案分析')

    df_bc_ana, build_case, color_by, c = fn_gen_analysis_sel(df.copy(), build_case, latest_records, key='bc_ana',
                                                          sel_option=['dist', 'build_case'])
    df_sel = df_bc_ana[df_bc_ana['建案名稱'] == build_case]
    df_sel.rename(columns={'MRT': '捷運站', 'MRT_DIST': '捷運站距離(m)'}, inplace=True)

    # st.write(df_sel)

    tab_bc_sell_2, tab_bc_sell_1, tab_bc_location, tab_bc_info = st.tabs(['銷售整理', '銷售明細', '建案位置', '建案資料'])

    with tab_bc_location:
        if build_case == '不限':
            st.subheader('🗺️ 建案位置')
        else:
            st.subheader(f'🗺️ 建案位置')
            st.write(f'- 建案地址: {df_sel["地址"].values[0]}')
            st.write(
                f'- 鄰近小學: {df_sel["sku_name"].values[0]} (距離: {int(df_sel["sku_dist"].values[0])}公尺, 學生人數: {int(df_sel["sku_109_total"].values[0])}人)')
            st.write(
                f'- 鄰近捷運: {df_sel["捷運站"].values[0]} (距離: {int(df_sel["捷運站距離(m)"].values[0])}公尺, 通勤時間: {int(df_sel["MRT_Commute_Time_UL"].values[0])}分鐘)')
            # st.write(f'- 通勤時間: {int(df_sel["MRT_Commute_Time_UL"].values[0])}分鐘 (捷運)')

        df_sel['每坪單價'] = df_sel['每坪單價(萬)'].apply(lambda x: str(x) + '萬/坪')

        title = ''
        hover_name = '建案名稱'
        hover_data = ['交易年', '總價(萬)', '每坪單價(萬)', '車位單價(萬)',
                      '車位類別', '移轉層次', '捷運站', '捷運站距離(m)', ]
        map_style = "open-street-map"
        fig_map = fn_gen_plotly_map(df_sel, title, hover_name, hover_data, map_style, zoom=14)

        # if build_case != '不限':
        #     bc_dist = df_sel[df_sel['建案名稱'] == build_case]['鄉鎮市區'].values[0]
        #     bc_vill = df_sel[df_sel['建案名稱'] == build_case]['里'].values[0]
        #     k = f'{"臺北市"}, {bc_dist}, {bc_vill}'
        #     fig_map = fn_plot_shp(fig_map, dic_of_shp, k, text=k)

        fn_dbg('fn_gen_web_eda 3-2')
        st.plotly_chart(fig_map)
        st.write('')

        fn_dbg('fn_gen_web_eda 4')

    with tab_bc_info:
        st.subheader('📜 建案資料')
        if build_case == '不限':
            st.markdown(f'{"#" * 4} 🙊 請選擇 👉 建案名稱')
        else:
            # st.subheader(f'🚇 捷運 {mrt.split("_")[-1]} 周邊 👉 {build_case}')
            # st.write('')

            builder = 'NA'
            builders = [builder]
            constructor = 'NA'
            constructors = [constructor]

            df_lg = pd.read_csv(os.path.join(dic_of_path['database'], 'builder_litigation.csv'), na_filter=False,
                                encoding='utf-8-sig')

            with st.form(key='Form_bc_info'):
                c1, c2 = st.columns(2)
                bc_info_c1 = ['建案名稱', '投資建設', '營造公司', '建造執照', '完工年度', '地上樓層', '地下樓層', '總戶數', '企劃銷售']
                bc_info_c2 = ['基地面積(坪)', '建蔽面積(坪)', '建蔽率(%)', '容積率(%)', '公設比(%)', '平面車位', '機械車位', '座向規劃', '土地分區']

                for i in bc_info_c1:
                    v = str(df_sel[i].values[0])
                    v = v.split('.')[0] if '總戶數' in i or '車位' in i or '面積' in i else v
                    v = v + '%' if '%' in i else v
                    v = v + '年' if i == '完工年度' else v
                    if i == '建造執照' and v != 'nan':
                        url = r'https://tccmoapply.dba.tcg.gov.tw/tccmoapply/maliapp/asp/aspcons01.do?node=20181011170807773001'
                        v = f'[{v}]({url})'
                    c1.write(f'{i}: {v}')
                    builder = v if i == '投資建設' else builder
                    builders = fn_util_split(builder) if i == '投資建設' else builders
                    constructor = v if i == '營造公司' else constructor
                    constructors = fn_util_split(constructor) if i == '營造公司' else constructors

                    for builder in builders:
                        if i == '投資建設' and builder in df_lg['建商營造'].values:
                            df_lg_b = df_lg[df_lg['建商營造'] == builder]
                            lg_latest = df_lg_b['裁判日期'].values[0]
                            lg_reason = df_lg_b['裁判案由'].values[0]
                            b = ''  # if len(builders) <= 1 else f' ({builder})'
                            c1.write(
                                f'👉 最新裁判: [{lg_latest} - {lg_reason}](https://judgment.judicial.gov.tw/FJUD/default.aspx) ❗{b}')

                    for constructor in constructors:
                        if i == '營造公司' and constructor in df_lg['建商營造'].values:
                            df_lg_b = df_lg[df_lg['建商營造'] == constructor]
                            lg_latest = df_lg_b['裁判日期'].values[0]
                            lg_reason = df_lg_b['裁判案由'].values[0]
                            c = ''  # if len(constructors) <= 1 else f' ({constructor})'
                            c1.write(
                                f'👉 最新裁判: [{lg_latest} - {lg_reason}](https://judgment.judicial.gov.tw/FJUD/default.aspx) ❗{c}')

                for i in bc_info_c2:
                    v = str(df_sel[i].values[0])
                    v = v.split('.')[0] if '總戶數' in i or '車位' in i or '面積' in i else v
                    v = v + '%' if '%' in i else v
                    v = v + '年' if i == '完工年度' else v
                    # v = '[參考公式](https://www.hbhousing.com.tw/News/Detail.aspx?Num=5044&utm_source=gsm&utm_medium=cpc&utm_content=word&utm_campaign=201901hbcue&ctype=B&cid=words&oid=gsm&gclid=Cj0KCQjw_7KXBhCoARIsAPdPTfhs4zRHqUrHu26rpRRl1RaWymm0iNi04nTTKiA9pNtWRqWQnmKutLwaAr_OEALw_wcB)' if i=='容積率(%)' else v
                    v = v + ' [法規](https://www.udd.gov.taipei/laws/rdpqpr5-5426)' if i == '容積率(%)' else v
                    c2.write(f'{i}: {str(v).replace(".0%", "%")}')

                submitted = st.form_submit_button("")

            st.write()
            st.subheader('  ⚖️訴訟裁判')
            if builder in df_lg['建商營造'].values:
                df_lg_b = df_lg[df_lg['建商營造'] == builder]
                df_lg_b = df_lg_b[['建商營造', '歷年案件', '裁判日期', '裁判案由', '裁判字號']]
                lg_latest = df_lg_b['裁判日期'].values[0]
                lg_total = df_lg_b['歷年案件'].values[0]

                with st.expander(f' 👉 建商:{builder} {"    "}  ⚖️歷史裁判案件數: {lg_total}件 ❗'):
                    st.write('')
                    st.write(f'- 資料來源: [司法院 法學資料檢索系統](https://judgment.judicial.gov.tw/FJUD/default.aspx)')
                    st.write('')
                    AgGrid(df_lg_b, theme='blue', enable_enterprise_modules=True)

            for c in constructors:
                # fn_dbg(c)
                if c in df_lg['建商營造'].values:
                    df_lg_c = df_lg[df_lg['建商營造'] == c]
                    df_lg_c = df_lg_c[['建商營造', '歷年案件', '裁判日期', '裁判案由', '裁判字號']]
                    lg_latest = df_lg_c['裁判日期'].values[0]
                    lg_total = df_lg_c['歷年案件'].values[0]

                    with st.expander(f'👉 營造商:{c} {"    "}  ⚖️ 歷史裁判案件數: {lg_total}件 ❗'):
                        st.write('')
                        st.write(f'- 資料來源: [司法院 法學資料檢索系統](https://judgment.judicial.gov.tw/FJUD/default.aspx)')
                        st.write('')
                        AgGrid(df_lg_c, theme='blue', enable_enterprise_modules=True)

        fn_dbg('fn_gen_web_eda 5')

    with tab_bc_sell_1:
        st.write('')
        st.subheader(f'{From_To}, 銷售速率 {round(len(df_sel["戶別"].unique()) / period, 2)} 筆/月')
        st.subheader(f'均價 {int(ave)} 萬/坪')

        dft_sel = ['移轉層次', '建物坪數', '每坪單價(萬)', '總價(萬)',
                   '車位類別', '車位單價(萬)', '交易年月日']

        if len(st.session_state['feature_sel']) == 0:
            st.session_state['feature_sel'] = dft_sel

        df_cols = df_sel[st.session_state['feature_sel']]
        with st.form(key='欄位選擇'):
            cols = st.multiselect(f'欄位選擇(共{len(df_sel.columns)}個)', df_sel.columns,
                                  default=st.session_state['feature_sel'])

            submitted = st.form_submit_button('選 擇')

            if submitted:
                df_cols = df_sel[cols]
                st.session_state['feature_sel'] = cols
                st.write(f'選擇了 {len(st.session_state["feature_sel"])}個欄位')

        df_cols = df_cols.sort_values(by='移轉層次', ascending=False) if '移轉層次' in df_cols.columns else df_cols

        AgGrid(df_cols, theme='blue', fit_columns_on_grid_load=True, enable_enterprise_modules=True)

        st.write(
            f'$資料來源:$ [$内政部不動產交易實價查詢服務網(每月1、11、21 日發布)$](https://plvr.land.moi.gov.tw/DownloadOpenData) $,\ 共{df_cols.shape[0]}筆資料$')

        fn_dbg(f'fn_gen_web_eda 6 {build_case}')

    with tab_bc_sell_2:
        if build_case == '不限':
            st.markdown(f'{"#" * 4} 🙊 請選擇 👉 建案名稱')
        else:
            # fn_gen_bc_deals(build_case, dic_df_show)

            dic_df_show = fn_gen_bc2show(df_sel)

            fn_dbg('fn_gen_web_eda 7')
            deals = np.count_nonzero(dic_df_show['每坪單價(萬)'])
            st.write('')
            # st.subheader(f'🏡 建案: {build_case}'
            #              f' 📝 登錄: {deals} 筆'
            #              f' 💰 總金額: {round((dic_df_show["總價(萬)"].values.sum()) / 10000, 2)} 億')
            st.markdown(f'{"#" * 4} '
                        f'🏡 建案: {build_case}'
                        f' 📝 登錄: {deals} 筆'
                        f' 💰 總金額: {round((dic_df_show["總價(萬)"].values.sum()) / 10000, 2)} 億')

            tabs = st.tabs(['每坪單價', '單價分布', '樓層價差', '總價-車位', '總價', '車位總價', '建物坪數', '車位坪數', '交易日期', '樓層均價'])

            tab_price, tab_price_dist, tab_diff, tab_wo_pk, tab_total, tab_pk, tab_area, tab_pk_area, tab_date, tab_floor_ave = tabs

            with tab_price:
                fn_gen_bc_deals(build_case, dic_df_show, '每坪單價(萬)')

            with tab_price_dist:
                fn_gen_bc_deals(build_case, dic_df_show, '單價分布')

            with tab_diff:
                fn_gen_bc_deals(build_case, dic_df_show, '樓層價差(%)')

            with tab_wo_pk:
                fn_gen_bc_deals(build_case, dic_df_show, '總價-車位(萬)')

            with tab_total:
                fn_gen_bc_deals(build_case, dic_df_show, '總價(萬)')

            with tab_pk:
                fn_gen_bc_deals(build_case, dic_df_show, '車位總價(萬)')

            with tab_area:
                fn_gen_bc_deals(build_case, dic_df_show, '建物坪數')

            with tab_pk_area:
                fn_gen_bc_deals(build_case, dic_df_show, '車位坪數')

            with tab_date:
                fn_gen_bc_deals(build_case, dic_df_show, '交易日期')

            with tab_floor_ave:
                fig_bar2 = fn_gen_floor_ave(df_sel)
                st.write('')
                st.plotly_chart(fig_bar2)


@fn_profiler
def fn_gen_web_ml_train(df, path):
    # ts = time.time()
    fn_dbg('fn_gen_web_ml_train 1')

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
        tune = col2.radio('調校方式', ('Manually', 'GridSearch (cv=5) 🐢', 'RandomizedSearch 🚧'), index=0)
        cv = int(tune.split('cv=')[-1].split(')')[0]) if 'cv=' in tune else 0
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
            if ml_model == 'RandomForestRegressor':
                dft_trees = 800
                dft_depth = 150
            else:
                dft_trees = 380
                dft_depth = 30

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

        for c in ['地下樓層', '總戶數', '基地面積(坪)']:
            df = df[df[c].apply(lambda x: str(x).split('.')[0].isnumeric())]
            assert df.shape[0] > 0, f'{c} df.shape = {df.shape}'
            df[c] = df[c].apply(lambda x: int(round(float(x), 0)))

        df.reset_index(drop=True, inplace=True)

        fn_dbg('fn_gen_web_ml_train 2')
        X, df_cat = fn_gen_training_data(df, path)
        fn_dbg('fn_gen_web_ml_train 3')
        y = df[['每坪單價(萬)']]

        with st.form(key='Form2'):
            st.markdown('##### 訓練特徵選擇:')
            features_sel = st.multiselect('特徵選擇:', X.columns,
                                          default=[c for c in X.columns if
                                                   '建材' not in c and
                                                   '車位類別' not in c and
                                                   'MRT_ave' not in c and
                                                   'DIST_ave' not in c and
                                                   'SKU_ave' not in c and
                                                   '稅_第' not in c and
                                                   '幾房' not in c and
                                                   '幾衛' not in c and
                                                   # '幾廳' not in c and
                                                   # '交易年' not in c and
                                                   # '使用分區' not in c and
                                                   # '利率' not in c and
                                                   # '中位數' not in c and
                                                   # '頂樓-1' not in c and
                                                   c != 'MRT'])
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
    is_train = st.button('硬Train一發!')
    if is_train or st.session_state['Train'] == 'done':
        X = X[features_sel]
        if drop_sel == '包含':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=X.loc[:, 'MRT'])
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=df['cat'])

        del X

        with st.expander(f'👓 檢視 資料篩選'):
            if drop_sel != '包含':
                st.markdown(f'{"#" * 5} 排除分佈少於{limit * 100} % 的目標: 共{drop_num}筆')

            if '無' not in ano_det:
                st.markdown(f'{"#" * 5} {ano_det} 的資料: 共{df_ano.shape[0]}筆')
                df_screen = df_ano[['MRT', '地址', '每坪單價(萬)', '交易年月日', '備註']]
                AgGrid(df_screen, theme='blue', enable_enterprise_modules=True)

        with st.expander(f'👓 檢視 資料分佈'):
            watch = "每坪單價(萬)"
            st.markdown(f'{"#" * 5} 預測目標 *“{watch}"* 在 訓練 與 測試 樣本的分佈狀況:')
            fig = make_subplots(rows=2, cols=1)

            margin = dict(t=10, b=0, l=0, r=0)
            fn_gen_plotly_hist(fig, y_train[watch], '訓練', row=1, margin=margin, width=500)
            fn_gen_plotly_hist(fig, y_test[watch], '測試', row=2, margin=margin, width=500)
            st.plotly_chart(fig)

            if 'MRT' in X_train.columns:  # X.columns:
                for col_2_check in ['MRT']:  # ,移轉層次,交易年
                    st.markdown(f'{"#" * 5} 特徵 *“{col_2_check}”* 在 訓練 與 測試 樣本的佈狀況:')
                    fig = make_subplots(rows=2, cols=1)
                    margin = dict(t=10, b=0, l=0, r=0)
                    fn_gen_plotly_hist(fig, X_train[col_2_check].sort_values(), f'訓練:{X_train.shape[0]} 筆', row=1,
                                       margin=margin, width=500)
                    fn_gen_plotly_hist(fig, X_test[col_2_check].sort_values(), f'測試: {X_test.shape[0]} 筆', row=2,
                                       margin=margin, width=500)
                    st.plotly_chart(fig)

        if 'MRT' in X_train.columns:  # X.columns:
            # X = X.drop(columns='MRT')
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
                    {'n_estimators': [500, 600, 800, 1000],
                     'max_depth': [50, 100, 150, 200]},
                    # ('bootstrap': [False],
                    # 'n_estimators': [150,200,259].
                    # 'max_features': [10, X_train.shape[1]]),
                ]

            regr = GridSearchCV(regr_sel,
                                param_grid,
                                cv=cv,
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

        try:
            regr.fit(X_train, y_train.values.ravel())
        except:
            df_dbg = X_train[X_train['交易年'] > 110]
            for c in df_dbg.columns:
                print(c)
                print(df_dbg[c])

            assert False, f'{X_train.shape, y_train.shape}'

        fn_dbg('fn_gen_web_ml_train 4')

        if tune == 'GridSearch':
            print(regr.best_params_)
            # st.write('')
            col2.text(f'{tune} Result:')
            col2.write(regr.best_params_)
            # st.write('')
            st.session_state['para'] = regr.best_params_

        st.session_state['Train'] = 'done'

        fn_gen_web_ml_eval(ml_model, model_file, regr, X_train, X_test, y_train, y_test, df, mse_th)

        fn_dbg('fn_gen_web_ml_train 5')
        # st.session_state['Train'] = 'done'

    # te = time.time()
    # dur = round(te - ts, 5)
    # print(f'fn_gen_web_ml_train: {dur} 秒')


@fn_profiler
def fn_gen_web_ml_eval(ml_model, model_file, regr, X_train, X_test, y_train, y_test, df, mse_th):
    # ts = time.time()
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
    mse = round(df_result.loc["MSE", "測試集"], 2)

    st.write('')
    # is_model_save = st.button('訓練並儲存 模型')
    if True:  # is_model_save:
        df_F = pd.DataFrame()
        df_F['Features'] = X_train.columns
        # df_F.to_csv(model_file.replace('.sav', '.csv'), encoding='utf-8-sig', index=False)
        # pickle.dump(regr, open(model_file, 'wb'))
        # mse = round(df_result.loc["MSE", "測試集"], 2)
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

            # Remove existed models before saving new model ( save memory space in cloud side )
            model_fdr = os.path.dirname(model_file)
            for i, j, files in os.walk(model_fdr):
                for f in files:
                    if model_typ in f:
                        remove_file = os.path.join(model_fdr, f)
                        os.remove(remove_file)
                        print(f'{remove_file} removed !')

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

    margin = dict(t=30, b=0, l=0, r=400)
    fig = fn_gen_plotly_hist(fig, df_metrics['誤差(萬/坪)'], '測試誤差分佈(萬)', margin=margin, opacity=0.7, height=220)
    fig = fn_gen_plotly_hist(fig, df_sel['誤差(萬/坪)'], '測試誤差分佈(萬)', margin=margin, bins=10, barmode='overlay',
                             opacity=0.7, height=220)

    st.write('')
    c2.plotly_chart(fig)

    X_train.rename(columns={'sku_dist': '小學距離',
                            'sku_total': '鄰近小學人數',
                            'MRT_DIST': '捷運距離',
                            'MRT_Tput_UL': '捷運進站人數(上班)',
                            'MRT_Tput_DL': '捷運出站人數(上班)',
                            'MRT_Tput': '捷運人流(上班)',
                            'MRT_Commute_Time_UL': '捷運通勤時間',
                            'MRT_ave': '鄰近捷運<br>區域均價',
                            'DIST_ave': '行政區<br>區域均價',
                            'SKU_ave': '鄰近小學<br>區域均價',
                            '頂樓-1': '次頂樓',
                            '移轉層次': '樓層',
                            '稅_中位數': '所得中位數',
                            '稅_平均數': '所得平均數',
                            '稅_第一分位數': '所得第一分位數',
                            '稅_第三分位數': '所得第三分位數'}, inplace=True)

    try:
        df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': regr.feature_importances_})
    except:
        df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': regr.best_estimator_.feature_importances_})

    df_imp = df_imp.sort_values(by='Importance')

    imp_thd = 0.005
    # df_top = df_imp.iloc[df_imp.shape[0] - 10:df_imp.shape[0] + 1, :]
    # df_bot = df_imp.iloc[:10, :]
    df_imp['Features'] = df_imp['Features'].apply(lambda x: str(x) + '  ')
    df_imp['Importance'] = df_imp['Importance'].apply(lambda x: round(x, 5))
    df_top = df_imp[df_imp['Features'].apply(lambda x: '均價' in x)]
    df_bot = df_imp[df_imp['Features'].apply(lambda x: '均價' not in x)]
    df_bot = df_bot[df_bot['Importance'] > imp_thd]

    x_data_col = 'Importance'
    y_data_col = 'Features'
    color_col = 'Importance'
    text_col = 'Importance'
    v_or_h = 'h'
    margin = dict(t=0, b=0, l=10, r=0)
    text_fmt = '%{value:.5f}'

    if df_top.shape[0] > 0:
        fig_top = fn_gen_plotly_bar(df_top, x_data_col, y_data_col, text_col, v_or_h, margin,
                                    color_col=color_col, text_fmt=text_fmt, op=0.8,
                                    x_title='重要度 (影響力)', y_title='', height=300, font_size=13)

        c1, c2, c3 = st.columns(3)
        c2.markdown(f'{"#" * 6} 區域均價 對 房價 的影響')
        st.plotly_chart(fig_top)

    fig_bot = fn_gen_plotly_bar(df_bot, x_data_col, y_data_col, text_col, v_or_h, margin,
                                color_col=color_col, text_fmt=text_fmt, ccs='haline', op=0.8,
                                x_title='重要度 (影響力)', y_title='', height=550, font_size=13)
    c1, c2, c3 = st.columns([1.5, 5, 0.5])
    model = ml_model.replace('Regressor', '') if 'Regressor' in ml_model else ml_model
    c2.markdown(
        f'{"#" * 6} 各項指標(Top {df_bot.shape[0]}) 對 房價 的影響力 ({model} MSE={round(df_result.loc["MSE", "測試集"], 2)})')
    st.plotly_chart(fig_bot)

    st.write('測試資料集 的 模型預估結果(萬/坪):')
    # st.dataframe(df_metrics)
    AgGrid(df_metrics, theme='blue', enable_enterprise_modules=True)

    # te = time.time()
    # dur = round(te - ts, 5)
    # print(f'fn_gen_web_ml_eva:{dur}秒')


@fn_profiler
def fn_gen_web_ml_inference(path, build_typ):
    # ts = time.time()

    ml_model = os.path.join(path, r'output/model')
    if not os.path.exists(ml_model):
        os.makedirs(ml_model)

    # model_file = os.path.join(ml_model, 'ml_model.sav')

    model_fdr = ml_model
    models = []
    dates = []
    for i, j, files in os.walk(ml_model):
        for f in files:
            if '.sav' in f and f not in models:
                models.append(f)
                latest = f.split('.sav')[0].split('_')[-1]
                try:
                    dates.append(int(latest))
                except:
                    print(f'date parsing fail ! -->  {latest}')

    # keep = dates.index(max(dates))

    dic_keep = {}
    dates.sort(reverse=True)
    for d in dates:
        m_id = [i for i, n in enumerate(dates) if n == d]

        for i in m_id:
            m = models[i]

            if '_rf_' in m and 'rf' not in dic_keep.keys():
                dic_keep['rf'] = m

            if '_xgb_' in m and 'xgb' not in dic_keep.keys():
                dic_keep['xgb'] = m

        print(d, dic_keep)

    for m in models:
        if m not in dic_keep.values():
            print(f' {m} not in {dic_keep.values()} remove it !')
            os.remove(os.path.join(ml_model, m))
            os.remove(os.path.join(ml_model, m.replace('.sav', '.csv')))

        # drop = os.path.join(ml_model, m)
        # os.remove(drop) if models.index(m) != keep else None
        # print(models.index(m) != keep, m, keep, dates[keep], drop, dates, models)

    if len(models) > 0:
        st.write('')
        # st.subheader('模型推論')
        st.markdown('#### :blue[$模型選擇:$]')
        model_sel = st.selectbox('模型選擇:', models, label_visibility='collapsed')
        model_typ = model_sel.split('tpe')[-1].split('mse')[0].replace('_', '')
        model_sel = os.path.join(model_fdr, model_sel)

        # load the model from disk
        loaded_model = fn_load_model(model_sel)

        df_F = pd.read_csv(model_sel.replace('.sav', '.csv'), encoding='utf-8-sig')

        dic_of_input = {}
        with st.form('Form2'):
            c1, c2, c3 = st.columns([1, 1, 2])

            tpe_dists = ['中正區', '大同區', '中山區', '松山區', '大安區', '萬華區',
                         '信義區', '士林區', '北投區', '內湖區', '南港區', '文山區']

            input_city = c1.selectbox('城市', ['台北市'], index=0)
            input_dist = c2.selectbox('行政區', tpe_dists, index=tpe_dists.index('北投區'))
            input_addr = c3.text_input(label='詳細地址', value='大度路三段301巷67號')
            addr = input_city + input_dist + input_addr

            # addr = st.text_input(label='物件地址', value='台北市北投區大度路三段301巷67號')

            addr = fn_addr_handle(addr)
            df_coor_read = fn_house_coor_read()

            # build case = fn_addr_2_build_case(addr)

            geo_info, is_coor_save, is_match, addr_fr_db = fn_get_geo_info(addr, df_coor_read, slp=5)

            if addr in df_coor_read.index:
                vill = df_coor_read.loc[addr, '里']
                # st.write(f'鄰近地址: {is_match} {addr} {vill}')
            elif is_match:
                vill = df_coor_read.loc[addr_fr_db, '里']

                st.write(f'鄰近地址: {is_match} {addr_fr_db} {vill}')
            else:
                print(addr)
                assert False, f'addr error: {addr} ToDo: Add vill from addr !'

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
                    print(f'save {addr}')
                    fn_house_coor_save(df_coor_save)

            dic_of_input['台北市'] = 1 if '台北市' in addr else 0

            for d in geo_info.keys():
                for k in geo_info[d].keys():
                    dic_of_input[k] = geo_info[d][k]

            ave_path = dic_of_path['database']
            df_sku_ave = pd.read_csv(os.path.join(ave_path, 'SKU_ave.csv'), index_col='sku_name')
            df_mrt_ave = pd.read_csv(os.path.join(ave_path, 'MRT_ave.csv'), index_col='MRT')
            df_dist_ave = pd.read_csv(os.path.join(ave_path, 'DIST_ave.csv'), index_col='鄉鎮市區')
            df_tax = pd.read_csv(os.path.join(ave_path, '108_165-A.csv'), index_col='行政區')

            mrt = dic_of_input['MRT']
            dic_of_input['MRT_ave'] = df_mrt_ave.loc[mrt, '每坪單價(萬)']

            sku = dic_of_input['sku_name']
            sku = sku if sku in df_sku_ave.index else fn_get_neighbor(sku, df_sku_ave.index)
            dic_of_input['SKU_ave'] = df_sku_ave.loc[sku, '每坪單價(萬)']
            dist = addr.split('市')[-1].split('區')[0] + '區'
            dic_of_input['DIST_ave'] = df_dist_ave.loc[dist, '每坪單價(萬)']

            df_tax = df_tax[df_tax.index == dist]
            dic_of_input['稅_中位數'] = df_tax[df_tax['里'] == vill]['中位數'].values[0]
            dic_of_input['稅_平均數'] = df_tax[df_tax['里'] == vill]['平均數'].values[0]

            dic_of_input['緯度'] = dic_of_input.pop('lat')
            dic_of_input['經度'] = dic_of_input.pop('log')

            c1, c2, c3, c4 = st.columns(4)
            dic_of_input['總戶數'] = c1.text_input(label='總戶數', value=60)
            dic_of_input['基地面積(坪)'] = c2.text_input(label='基地面積(坪)', value=332)
            dic_of_input['建物坪數'] = c3.text_input(label='建物坪數(不含車位):', value=24)
            dic_of_input['車位坪數'] = c4.text_input(label='車位坪數:', value=2.21)

            c1, c2, c3, c4 = st.columns(4)
            this_yr = datetime.date.today().year - 1911
            dic_of_input['交易年'] = c1.slider('交易年(民國)', min_value=100, max_value=120, step=1, value=this_yr)
            dic_of_input['交易月'] = datetime.date.today().month
            dic_of_input['移轉層次'] = c2.slider('交易樓層', min_value=2, max_value=40, step=1, value=14)
            dic_of_input['總樓層數'] = c3.slider('地上樓層', min_value=2, max_value=40, step=1, value=15)
            # dic_of_input['屋齡'] = c4.slider('屋齢', min_value=0, max_value=20, step=1, value=0)
            dic_of_input['地下樓層'] = c4.slider('地下樓層', min_value=0, max_value=7, step=1, value=2)

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

    st.write(" ")
    st.subheader(f'其它房價預測平台:')
    st.markdown(
        f'{"#" * 4} 🔮 [:blue[$中信銀行-智慧估價平台$]](https://www.ctbcbank.com/content/dam/minisite/long/loan/ctbc-mortgage/index.html)')
    st.markdown(f'{"#" * 4} 🔮 [:blue[$好時價$]](https://www.houseplus.tw/)')

    # te = time.time()
    # dur = round(te - ts, 5)
    # print(f'fn_gen_web_inference: {dur} 秒')


@fn_profiler
def fn_gen_web_init(path, page=None):
    # print('fn_gen_web_init start')
    path_output = os.path.join(path, r'output')
    path_output = os.path.join(path_output, r'house_all.csv')
    # print(path_output)
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

    if 'feature_sel' not in st.session_state:
        st.session_state['feature_sel'] = []

    # print(f'session_state: {st.session_state}')
    df = fn_get_house_data(path_output)
    df = fn_cln_house_data(df.copy())

    if page == 'train':
        cat_features = ['鄉鎮市區', '主要建材', '車位類別', 'MRT']
        for cat in cat_features:
            df_cat = pd.DataFrame(columns=[cat], data=sorted(list(df[cat].unique())))
            file = os.path.join(path, f'output\\Feature_{cat}.csv')
            df_cat.to_csv(file, encoding='utf-8-sig')

    # print('fn_gen_web_init done')
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
    st.write("- 經濟成長: [行政院主計總處 - 中華民國統計資訊網](https://www.stat.gov.tw/point.asp?index=1)")
    st.write("- 人口普查: [行政院主計總處 - 109年普查初步統計結果表](https://www.stat.gov.tw/ct.asp?mp=4&xItem=47698&ctNode=549)")
    st.write("- 單點坐標回傳行政區: [政府資料開放平台 - 坐標回傳行政區API](https://data.gov.tw/dataset/101898)")
    st.write("- 鄉鎮市區界線: [政府資料開放平台 - 我國各鄉(鎮、市、區)行政區城界線圖資](https://data.gov.tw/dataset/441)")
    st.write("- 村里界圖: [政府資料開放平台 - 各縣市村(里)界](https://data.gov.tw/dataset/7438)")
    st.write("- 所得分析: [政府資料開放平台 - 綜稅所得鄉鎮村里統計分析表](https://data.gov.tw/dataset/17983)")
    st.write(
        "- 所得分析: [表165-A(108年度)綜稅所得總額各縣市鄉鎮村里統計分析表](https://www.fia.gov.tw/WEB/fia/ias/isa108s/isa108/108_165-A.pdf)")

    st.write("")
    st.subheader('參考網站:')
    st.write("- 實價登錄網站: [樂居](https://www.leju.com.tw/)")
    st.write("- 實價登錄網站: [實價登錄比價王](https://community.houseprice.tw/building/118031)")
    st.write("- 實價登錄網站: [PLEX 專注大台北大廈的房屋網](https://www.plex.com.tw/)")
    st.write(
        "- 房價預測網站: [中信銀行 智慧估價平台](https://www.ctbcbank.com/content/dam/minisite/long/loan/ctbc-mortgage/index.html)")
    st.write("- 房價預測網站: [好時價House+(利用統計學、數學及人工智慧(AI)演算法,算出不動產價值)](https://www.houseplus.tw/)")
    st.write("- 房價指數: [國立清華大學 安富金融工程研究中心](https://aife.site.nthu.edu.tw/p/404-1389-220340.php)")
    st.write("- 房價指標: [臺北市地政局 房市指標溫度計](https://emuseum.land.gov.taipei/TaipeiLandRealEstate/)")
    st.write("- 裁判書查詢: [司法院 法學資料檢索系統](https://judgment.judicial.gov.tw/FJUD/default.aspx)")
    st.write("- 耐震標章: [台灣建築中心 耐震標章](https://www.tabc.org.tw/sab/modules/news/article.php?storyid=7)")

    st.write('')
    st.subheader('相關競賽:')
    st.write(
        "- 交通部: [交通數據創新應用競賽](https://tdx-contest.tca.org.tw) [TDX交通資料育成網](https://startup.transportdata.tw/) [隊名: 傑克潘 (TD-81670023)](https://tdx-contest.tca.org.tw/)")
    st.write("- 玉山人工智慧公開挑戰賽2019夏季賽: [台灣不動產AI神預測](https://tbrain.trendmicro.com.tw/competitions/Details/6)")
    st.write("- 經濟部中小企業處: [2021城市數據實境賽](https://data.startupterrace.tw/data-contest)")

    st.write("")
    st.subheader('專利:')
    st.write("- 智能不動產估價專利: [中華民國專利資訊檢索系統](https://twpat2.tipo.gov.tw/twpatc/twpatkm?@@642176895)")

    st.write('')
    st.subheader('當你要買預售屋...')
    with st.expander('簽約之前 的 注意事項'):
        st.write('')
        st.subheader('建商/營造商 的 規模與履歷? 一案建商?')
        st.write("- 經濟部 商業司: [商工登記公示資料查詢服務](https://findbiz.nat.gov.tw/fts/query/QueryBar/queryInit.do)")
        st.write("- 內政部 營建署: [建築工程履歷查詢系統](http://cpabm.cpami.gov.tw/cers/SearchLicForm.do)")
        st.write("- 透明足跡: [掃了再買－讓企業負起責任](https://thaubing.gcaa.org.tw/)")
        st.write(
            "- 台北市建管業務: [施工進度案件查詢](https://tccmoapply.dba.tcg.gov.tw/tccmoapply/maliapp/asp/aspcons01.do?node=20181011170807773001)")

        st.write('')
        st.subheader('廣告不實 怎麼辦?')
        st.write(
            "- 台北市 地政局: [地權及不動產交易科](https://land.gov.taipei/News_Content.aspx?n=8C8F186F23B3BE43&sms=1EA0BE6515958939&s=88696428E9FB14CA)")
        st.write("- 內政部 地政司: [不動產交易管理科](https://www.land.moi.gov.tw/chhtml/mailbox/54)")
        st.write("- 行政院 公平會: [服務信箱](https://www.ftc.gov.tw/internet/main/mailbox/notice.aspx)")
        st.write("- 行政院 消基會: [線上申訴調解申請](https://appeal.cpc.ey.gov.tw/WWW/Default.aspx)")

        st.write('')
        st.subheader('訴訟案件 哪裡查?')
        st.write("- 司法院 法學資料檢索系統: [裁判書查詢](https://judgment.judicial.gov.tw/FJUD/default.aspx)")

        st.write('')
        st.subheader('建築標章 哪裡查?')
        st.write("- 財團法人台灣建築中心: [建築標章](https://www.tabc.org.tw/tw/)")

        st.write('')
        st.subheader('都更規劃 哪裡查?')
        st.write("- 內政部 營建署: [都更查詢](https://twur.cpami.gov.tw/zh/urban/rebuild/view/621)")
        st.write("- 台北市 都更處: [都市更新處](https://uro.gov.taipei/)")
        st.write("- 財訊: [都更全都通](https://www.urbanrenewal.wealth.com.tw/)")

        st.write('')
        st.subheader('土地使用分區 哪裡查?')
        st.write("- 台北市 工務局: [臺北地理資訊e點通](https://addr.gov.taipei/M2019/indexPwd.aspx)")
        st.write("- 台北市 都發局: [臺北市歷史圖資展示系統](https://www.historygis.udd.gov.taipei/urban/)")

        st.write('')
        st.subheader('地質條件 哪裡查?')
        st.write("- 經濟部 中央地質研究所: [工程地質探勘資料庫](https://www.moeacgs.gov.tw/)")
        st.write("- 行政院 國家災害防救科技中心: [災害潛勢地圖](https://dmap.ncdr.nat.gov.tw/)")

        st.write('')
        st.subheader('工安意外 哪裡查?')
        st.write("- 勞動部 職業安全署: [重大職業災害公開網](https://pacs.osha.gov.tw/17238)")

        st.write('')
        st.subheader('其他')
        st.write(
            "- 內政部 營建署: [建築執照申請審核書](https://www.cpami.gov.tw/%E6%9C%80%E6%96%B0%E6%B6%88%E6%81%AF/%E6%B3%95%E8%A6%8F%E5%85%AC%E5%91%8A/30-%E5%BB%BA%E7%AF%89%E7%AE%A1%E7%90%86%E7%AF%87/28578-%E5%BB%BA%E7%AF%89%E5%9F%B7%E7%85%A7%E7%94%B3%E8%AB%8B%E5%AF%A9%E6%A0%B8%E6%9B%B8%E8%A1%A8.html)")
        st.write(
            "- 台北市 建管處: [建照執照申請表](https://dba.gov.taipei/News_Content.aspx?n=5B651B337CE7F386&sms=59F8DF70DEAE0B38&s=162C96AA9A55DB66)")


def fn_gen_web_tools():
    st.write("")
    st.subheader('機器學習:')
    st.write(
        "- 教科書: [Hands on Machine Learning - 第二章: 美國加州房價預測](https://nbviewer.org/github/DeqianBai/Hands-on-Machine-Learning/blob/master/02_Housing.ipynb)")
    st.write(
        "- 碩士論文: [淡江大學碩士在職專班 應用人工智慧於房價預測模型研究與分析(2019)](https://etds.lib.tku.edu.tw/ETDS/Home/Detail/U0002-2608201910580000)")

    st.write('')
    st.subheader('網頁製作:')
    st.write("- 純Python的極速網頁製作套件: [Streamlit](https://streamlit.io/)")
    st.write(
        "- Streamlit multi page framework: [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps)")
    st.write("- 畫文字 表情符號: [Emojipedia](https://emojipedia.org/)")
    st.write("- 影音嵌入: [Streamlit-player](https://github.com/okld/streamlit-player)")
    st.write("- 音樂庫: [SoundCloud](https://soundcloud.com/)")

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
    st.write("- 圖轉CSV: [誠華 OCR](https://zhtw.109876543210.com/)")


def fn_gen_web_projs():
    st.write('')
    st.subheader('📌 專案: 🅱️ [$B 計畫$](https://tw-stock.streamlit.app/)')
    st.subheader('📌 專案: 🌏 [$座標查詢行政區$](https://share.streamlit.io/ssp6258/use_conda_env/GeoPandas.py)')
    st.subheader('📌 專案: 🎲 [$離散事件模擬器$](https://share.streamlit.io/ssp6258/des_app/app.py)')


def fn_gen_web_intro():
    st.markdown('''

    ##### 這是一個專注於  **"台北市 預售屋交易"**  的數據分析網站 ~
    * 預售屋 **交易紀錄**
    * 預售屋 **銷售分析**
    * 預售屋 **房價預測**

    ''', unsafe_allow_html=True)

    tabs = st.tabs(['開發動機', '網站導覽', 'AI、機器學習、深度學習 原理及應用', '機器學習專案流程介紹', '與我聯絡'])
    tab_why, tab_intro, tab_ml, tab_ml_flow, tab_contact = tabs

    st.write('')
    with tab_why:  # st.expander('📌 開發動機'):
        st.write('')
        st.subheader('💡 對數據分析有股莫名的興趣 ~ ')
        st.subheader('💡 整理目前習得的技法, 應用於生活場景中 ~')
        st.write('')

    with tab_intro:  # st.expander('📌 網站導覽'):
        st.write('')
        st.subheader('🚧 晚點再寫 ...')
        st.write('')

    with tab_ml:  # st.expander('📌 AI、機器學習、深度學習 原理及應用'):
        st.write('')
        st.write(
            '- [十三分鐘略懂 AI 技術：機器學習、深度學習技術原理及延伸應用](https://youtu.be/UGdG4WpluJ8?list=PLySGbWJPNLA8D17qZx0KVkJaXd3qxncGr)')
        st.write('')
        video = 'https://www.youtube.com/watch?v=UGdG4WpluJ8'
        try:
            st_player(video, key=str(random.randint(0, 1000)), playing=False, loop=True, volume=0.5)
        except:
            pass

    with tab_ml_flow:  # tab_ml_flow:  # st.expander('📌 機器學習專案流程介紹'):
        st.write('')
        st.write(
            "- 引用自 Medium - Towards Data Science: [Workflow of a Machine Learning project - Ayush Pant](https://towardsdatascience.com/workflow-of-a-machine-learning-project-ec1dba419b94)")

        dic_of_img = {
            'ML flow': ['Overview of ML workflow', 'https://miro.medium.com/max/963/1*QV1rVgh3bfaMbtxueS-cgA.png'],
            'ML models': ['Overview of models under categories',
                          'https://miro.medium.com/max/656/1*KFQI59Yv7m1f3fwG68KSEA.jpeg'],
            'SciKit Learn': ['https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html',
                             'https://scikit-learn.org/stable/_static/ml_map.png'],
            'Unsupervise': ['Unsupervised Learning - Clustering',
                            'https://miro.medium.com/max/912/1*NjaQylKN3GUJGLGcdcgHlQ.png'],
            'Supervise1': ['Supervised Learning- Classification',
                           'https://miro.medium.com/max/963/1*PQ8tdohapfm-YHlrRIRuOA.gif'],
            'Supervise2': ['Supervised Learning - Regression',
                           'https://miro.medium.com/max/963/1*0Ve21Rildq950wRrlJvdLQ.gif'],
            'Train Test Split': ['Train Test Split', 'https://miro.medium.com/max/963/1*CeALK-1lzIWNJ7wN9DStlw.png'],
            'ELT': ['Data ELT(Extra Load Transform) for MNIST dataset',
                    'https://raw.githubusercontent.com/profundo-lab/imagenes/master/uPic/e6dLOp.png'],
            'MNIST1': ['MNIST using Deep Learning (ANN)', 'https://miro.medium.com/max/1160/0*u5-PcKYVfUE5s2by.gif'],
            'MNIST2': ['MNIST using Machine Learning (Random Forest)',
                       'https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif'],
            'Confusion Matrix': [
                'Classification Metrics - Confusion Matrix (Accuracy/Precision/Recall/F1-score/AUC/ROC)',
                'https://media.geeksforgeeks.org/wp-content/uploads/20200821144709/284.PNG'],
            'Regression Metrics': ['Regression Metrics - MAE/MSE/RMSE/...',
                                   'https://miro.medium.com/max/875/1*BFzp8uSMk88mDLibU465VA.png'],
        }

        st.write('')
        for k in dic_of_img.keys():
            fn_show_img('', dic_of_img[k][1], caption=dic_of_img[k][0])

    with tab_contact:  # st.expander('📌 與我聯絡'):
        st.write('')
        st.write("🔗 [Jack Pan](https://www.linkedin.com/in/%E9%95%B7%E6%B0%B8-%E6%BD%98-235912aa/)")
        st.write("📘 [Facebook](https://www.facebook.com/jack.pan.96/)")
        st.write('✉️ ssp6258@yahoo.com.tw')
        st.write('💾 [網站程式碼](https://github.com/SSP6258/house_app)')
        st.write('🚧 [故障報修、意見反饋](https://github.com/SSP6258/house_app/issues/new)')


def fn_chrome_96_workaround():
    # st.write('<style>div{font-weight: normal;}</style>', unsafe_allow_html=True)
    pass


dic_of_shp = {}
dic_of_shp_dist = {}


@st.cache
def fn_read_shp_wrap(is_dist_only=False):
    shapes, properties = fn_read_shp(is_dist_only=is_dist_only)

    shapes_new = {}
    properties_new = {}
    for k, v in properties.items():
        # print(f'{k}: {v}')
        shapes_new[v] = shapes[k]
        properties_new[v] = v

    return shapes_new, properties_new


def fn_add_style():
    # with open('style.css', encoding="utf-8") as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    pass


def fn_app(page='data'):
    # print(f'fn_app() start, page = {page}')
    fn_chrome_96_workaround()
    # st.legacy_caching.clear_cache()

    fn_add_style()

    this_yr = datetime.datetime.now().year - 1911
    dic_of_dbg['is_dbg'] = st.sidebar.checkbox('🔨️ 工程模式 ')
    st.sidebar.write('')
    dic_of_dbg['time'] = 0
    dic_of_dbg['sum'] = 0
    fn_dbg(f'', is_force=True)
    st.sidebar.markdown('### 🔍 :blue[$資訊篩選$]')
    year_sel = st.sidebar.slider('交易年(民國)', min_value=100, max_value=this_yr, value=(this_yr - 2, this_yr))
    price_sel = st.sidebar.slider('每坪單價(萬)', min_value=40, max_value=200, value=(40, 200))
    c1, c2 = st.sidebar.columns(2)
    sel = c1.selectbox('交易類別', ['預售屋', '中古屋'], index=0)
    root = dic_of_path['root']
    path = os.path.join(root, r'pre_sold_house') if sel == '預售屋' else os.path.join(root, r'pre_owned_house')
    ml_model = os.path.join(path, r'output\model')

    if not os.path.exists(ml_model):
        os.makedirs(ml_model)

    if page == 'eda':
        dic_of_shp['shape'], dic_of_shp['properties'] = fn_read_shp_wrap(is_dist_only=False)
        dic_of_shp_dist['shape'], dic_of_shp_dist['properties'] = fn_read_shp_wrap(is_dist_only=True)
        df = fn_gen_web_init(path)
        df = df[df['交易年'].apply(lambda x: year_sel[0] <= x <= year_sel[1])]
        df = df[df['每坪單價(萬)'].apply(lambda x: price_sel[0] <= x <= price_sel[1])]

        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)

        df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df

        c1, c2 = st.sidebar.columns(2)
        city = c1.selectbox('城市', ['台北市', '不限'], index=0)
        is_tpe = city == '台北市'
        df = df[df['台北市'] == is_tpe] if city != '不限' else df

        dists = ['不限'] + df['鄉鎮市區'].unique().tolist()
        d = c2.selectbox('鄉鎮市區', dists, index=dists.index('不限'))
        df = df[df['鄉鎮市區'] == d] if d != '不限' else df

        # land_typ = st.sidebar.selectbox('土地分區', ['不限', '住', '商'], index=0)
        # df = df[df['都市土地使用分區'] == land_typ] if land_typ != '不限' else df

        fn_gen_web_eda(df)

        # st.sidebar.write(properties)

    elif page == 'train':
        df = fn_gen_web_init(path, page=page)
        df = df[df['交易年'].apply(lambda x: year_sel[0] <= x <= year_sel[1])]
        df = df[df['每坪單價(萬)'].apply(lambda x: price_sel[0] <= x <= price_sel[1])]
        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)
        df = df[df['建物型態'] == build_typ] if build_typ != '不限' else df

        land_typ = st.sidebar.selectbox('土地分區', ['不限', '住', '商'], index=0)
        df = df[df['都市土地使用分區'] == land_typ] if land_typ != '不限' else df

        fn_gen_web_ml_train(df, path)
        for i in range(5):
            st.sidebar.write('')

    elif page == 'inference':
        build_typ = c2.selectbox('建物型態', ['大樓', '華廈', '不限'], index=0)
        fn_gen_web_ml_inference(path, build_typ)

        for i in range(5):
            st.sidebar.write('')

    elif page == 'reference':
        fn_gen_web_ref()

    elif page == 'tools':
        fn_gen_web_tools()

    elif page == 'projects':
        fn_gen_web_projs()

    elif page == 'introduce':
        fn_gen_web_intro()

    else:
        st.write(f' page: {page} unhandle yet !!!')

    print(f'fn_app() done, page = {page}')
    fn_dbg(f'fn_app end', is_force=True)
