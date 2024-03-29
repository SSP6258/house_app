import cn2an
import numpy as np
import re
import os
import datetime
import time
import pandas as pd
import pprint
from house_utils import fn_get_geo_info, fn_get_admin_dist, dic_of_path, fn_read_shp, fn_search_vill, fn_profiler, dic_bc_rename

dic_of_filter = {
    '交易標的': '房地(土地+建物)+車位',
    '主要用途': '住家用',
    '建物現況格局-隔間': '有',
}

dic_of_transfer = {
    '土地移轉總面積平方公尺': '土地移轉坪數',
    '建物移轉總面積平方公尺': '建物移轉坪數',
    '車位移轉總面積平方公尺': '車位移轉坪數',
    '車位移轉總面積(平方公尺)': '車位移轉坪數',
}

dic_of_typ = {
    '加強磚造': 'RC',
    '鋼筋混凝土造': 'RC',
    '鋼骨鋼筋混凝土造': 'SRC',
    '鋼骨結構': 'SC',
    '鋼骨造': 'SC',
    '鋼造': 'SC',
    '鋼骨混凝土造': 'SRC',
    '見其他登記事項': 'RC',
}

dic_of_name_2_addr = {
    # New TPE
    '勤樸丰硯': '新北市土城區明德路二段學士路口',
    '左岸玫瑰': '新北市八里區中山路一段308號',
    '令荷園': '新北市五股區新五路三段與新城八路',
    '鴻華天水': '新北市三重區環河北路三段490號',
    'Park188': '新北市淡水區濱海路二段202巷',
    '大隱小藍海': '新北市淡水區中正東路二段107巷3號',
    '馥樂樂': '新北市淡水區中山北路二段211號',
    '宜家': '新北市淡水區頂五路307號',
    # TPE
    'THE GREENHOUSE': '台北市中山區中山北路三段53號',
    '宏築信義': '台北市信義區和平東路三段463巷',
    '元利森活莊園': '台北市文山區木柵路一段209號',
    '昇陽逸居': '台北市萬華區中華路二段484巷',
    '鐫萃': '台北市松山區南京東路五段32號',
    '恆合江山': '台北市北投區承德路七段166號',
    '大安MONEY 賦寓': '台北市大安區羅斯福路三段155號',
    '大安MONEY 璞寓': '台北市大安區羅斯福路三段159號',
    '大安MONEY': '台北市大安區羅斯福路三段159號',
    '大喆': '台北市中山區民生東路二段115巷',
    '大?': '台北市中山區民生東路二段115巷',
    '日健闊': '台北市士林區重慶北路四段110號',
    '全坤X101': '台北市信義區信義路四段430號',
    '西園町': '台北市萬華區西園路一段36號',
    '四方荷韻': '台北市中正區重慶南路二段80號',
    '八方玥': '台北市南港區市民大道八段與南港路一段171巷',
    '天母常玉': '台北市北投區東華街一段550號',
    '吉美大安花園': '台北市大安區大安路二段80號',
    '吉美艾麗': '台北市松山區八德路四段480號',
    '首傅晴海': '台北市萬華區環河南路二段47號',
    '首傅晴海.': '台北市萬華區環河南路二段47號',
    '久康青沺': '台北市文山區木柵路三段85巷',
    '帝璽': '台北市中山區建國北路二段58號',
    '璞松綻': '台北市南港區市民大道七段8號',
    '達欣東門馥寓': '台北市中正區金山南路一段96號',
    '達永豐盛學': '台北市北投區中央北路一段88號',
    '璞玥': '台北市北投區致遠一路一段40號',
    '璞園寬心苑': '台北市台北市北投區石牌路二段307號',
    '松露院': '台北市秀明路二段75號',
    '敦北南京': '台北市松山區南京東路四段63號',
    '景美聯山-臻品區': '台北市文山區羅斯福路六段166巷',
    '遠雄晴川': '台北市文山區木柵路三段81號',
    '冠德羅斯福': '台北市中正區羅斯福路三段138號',
    '琢豐': '台北市中山區南京東路二段5號',
    '昭揚天際': '台北市北投區承德路七段140號',
    '維正里安': '台北市北投區立農街一段343巷',
    '宏國大道城B棟': '台北市大同區承德路二段101號',
    '至仁愛': '台北市中正區仁愛路二段74號',
    '景美聯山-極品區': '台北市文山區羅斯福路六段198號',
    '義泰吾境': '台北市內湖區南京東路六段389號',
    '華固大安學府': '台北市大安區復興南路二段37號',
    '基泰碧湖': '台北市內湖區內湖路一段741號',
    '華固文臨': '台北市北投區文林北路175號',
    '華固翡儷': '台北市北投區西安街二段197號',
    '明臣勸': '臺北市文山區指南路一段67號',
    '青耘上': '台北市文山區光輝路22巷',
    '國泰蒔美': '台北市內湖區行善路281巷',
    '忠泰湛': '臺北市文山區木新里開元街47號',
    '有川翩翩': '台北市士林區文林路594巷',
    '向陽上冠': '台北市中正區漢口街一段67號',
    '圓山帝寶': '台北市士林區承德路四段72號',
    '璞晛': '台北市北投區文林北路158號',
    '璞有聲-璞月': '台北市北投區西安街二段261號',
    '國泰華威豐年': '台北市北投區大業路520號',
    '當代一號院': '台北市大同區民族西路221巷33號',

}


def fn_sel_region(x):
    is_sel = False
    # if '淡水區' in x or '八里區' in x or '台北市' in x.replace('臺', '台'):
    if '台北市' in x.replace('臺', '台'):
        x = fn_addr_handle(x)
        is_sel = True
        if x.endswith('段'):
            print(x, '--> delete !')
            is_sel = False
    return is_sel


def fn_note_handle(x):
    bypass = ['親', '拆', '讓', '急', '減', '夾', '抵', '另']
    bypass += ['政府', '特殊', '利差', '和解', '無償', '裝潢', '家具', '家俱', '抽籤', '家電']
    bypass += ['地上權', '無土地', '樓中樓']
    bypass += ['非屬整戶', '分次登記']
    result = True

    if len(str(x)) == 0:
        return result

    for b in bypass:
        if b in str(x) and result:
            result = False
            return result

    return result


def fn_house_age_sel(x):
    try:
        result = int(x) > 900101
    except:
        print(f'Bypass Invalid Date format in 建築完成年月:{x}')
        result = False
    return result


@fn_profiler
def fn_house_filter(df):
    df = df.drop(index=0, columns=['編號', '非都市土地使用分區', '非都市土地使用編定'])
    df = df[df['都市土地使用分區'].apply(lambda x: str(x) == '住' or '住宅區' in str(x) or str(x) == '商')]

    if df.shape[0] == 0:
        return df

    # df = df[df['都市土地使用分區'].apply(lambda x: '商業區' not in str(x))]
    # df = df[df['主要建材'].apply(lambda x: '見其他登記事項' not in str(x))]
    df = df[df['移轉層次'].apply(lambda x: '，' not in str(x) and str(x) != 'nan')]  # 十四層， 十五層， 十三層

    df['交易年月日'] = df['交易年月日'].astype(int)
    file = df.loc[df.index[1], 'File'].lower()
    for idx in df.index:
        city = '未知'
        if file.startswith('f_'):
            city = '新北市'
        elif file.startswith('a_'):
            city = '台北市'
        else:
            print(f'Unknown city: {file}')

        if city != '未知' and df.loc[idx, '鄉鎮市區'] not in df.loc[idx, '土地位置建物門牌'] and \
                city not in df.loc[idx, '土地位置建物門牌'].replace('臺', '台'):
            df.at[idx, '土地位置建物門牌'] = df.loc[idx, '鄉鎮市區'] + df.loc[idx, '土地位置建物門牌']
            df.at[idx, '土地位置建物門牌'] = city + df.loc[idx, '土地位置建物門牌']

    df = df[df['土地位置建物門牌'].apply(lambda x: '地號' not in x)]

    if '建案名稱' in df.columns:
        df['建案名稱'] = df['建案名稱'].apply(lambda x: x.replace('.', '').replace('大?', '大喆').replace('吉美君?', '吉美君悅'))

    df_fix = df[df['土地位置建物門牌'].apply(lambda x: fn_addr_handle(x).endswith('段'))]

    if len(df['建築完成年月'].unique()) > 5:  # 中古屋
        df = df[df["建築完成年月"].apply(lambda x: True if len(str(x)) > 6 else False)]
        df = df[df['建築完成年月'].apply(fn_house_age_sel)]

    if '建案名稱' in df_fix.columns:
        for b in df_fix['建案名稱'].unique():
            if b in dic_of_name_2_addr.keys():
                for idx in df.index:
                    if df.loc[idx, ['建案名稱']].values == b:
                        df.at[idx, '土地位置建物門牌'] = dic_of_name_2_addr[b]
            else:
                bad_addr = df_fix[df_fix['建案名稱'] == b]['土地位置建物門牌'].values[0]
                is_tpe = '臺北市' in bad_addr or '台北市' in bad_addr
                if is_tpe:
                    print(is_tpe, bad_addr, b, '更新地址到 dic_of_name_2_addr{} !')

    df = df[df['土地位置建物門牌'].apply(fn_sel_region)]
    df = df[df['交易筆棟數'].apply(lambda x: False if '車位0' in x else True)]

    for k in dic_of_filter.keys():
        if k in df.columns:
            df = df[df[k] == dic_of_filter[k]]

    check = '建物型態'
    if check in df.columns:
        df = df[df[check].apply(lambda x: ('住宅大樓' in x) or ('華廈' in x))]

    df = df[df['車位移轉總面積(平方公尺)'].astype(float) > 0] if '車位轉總面積(平方公尺)' in df.columns else df

    if '備註' in df.columns:
        df = df[df['備註'].apply(fn_note_handle)]

    return df


@fn_profiler
def fn_data_cleaning(df):
    df = fn_house_filter(df)
    for k in dic_of_transfer.keys():
        if k in df.columns:
            df[dic_of_transfer[k]] = df[k].apply(lambda x: round(float(x) * 0.3025, 2))
            df = df.drop(columns=k)

    if df.shape == (0, 0):
        print(df.columns, df.shape, 'Empty df !!!')
        return df

    df['土地位置建物門牌'] = df['土地位置建物門牌'].apply(fn_addr_handle)
    df['地址'] = df['土地位置建物門牌'].apply(lambda x: x.split('北市')[-1].replace('旁', ''))
    df['台北市'] = df['土地位置建物門牌'].apply(lambda x: 1 if '台北市' in x.replace('臺', '台') else 0)
    df['戶別'] = df['棟及號'] if '棟及號' in df.columns else df['土地位置建物門牌'].apply(fn_get_house_hold)
    df['建物+車位移轉數'] = df['建物移轉坪數']
    df['建物移轉坪數'] = df['建物移轉坪數'] - df['車位移轉坪數']
    df['每坪單價'] = df['單價元平方公尺'].apply(lambda x: int(float(x) / 0.3025))

    df['主要建材'] = df['主要建材'].apply(lambda x: dic_of_typ[x] if x in dic_of_typ.keys() else x)
    df['每坪單價(萬)'] = df['每坪單價'].apply(lambda x: round(x / 10000, 2))
    df = df[df['每坪單價(萬)'] > 1.0]

    df['總價(萬)'] = df['總價元'].apply(lambda x: int(int(x) / 10000))
    df['車位總價(萬)'] = df['車位總價元'].apply(lambda x: int(int(x) / 10000))
    df['幾房'] = df['建物現況格局-房']
    df['幾廳'] = df['建物現況格局-廳']
    df['幾衛'] = df['建物現況格局-衛']
    df['車位'] = df['交易筆棟數'].apply(lambda x: x.split('車位')[-1])
    df['車位單價(萬)'] = round((df['車位總價元'].astype(int) / df['車位'].astype(int)) / 10000)
    df['車位單價(萬)'] = df['車位單價(萬)'].astype(int)

    df = df.drop(columns=['總價元',
                          '車位總價元',
                          '單價元平方公尺',
                          '有無管理組織',
                          '建物現況格局-房',
                          '建物現況格局-廳',
                          '建物現況格局-衛',
                          '交易筆棟數',
                          '每坪單價'])

    df['移轉層次'] = df['移轉層次'].apply(lambda x: cn2an.cn2an(str(x).replace('層', '')))
    df['總樓層數'] = df['總樓層數'].apply(lambda x: cn2an.cn2an(str(x).replace('層', '')) if type(x) == str and '層' in x else x)
    df['交易年'] = df['交易年月日'].apply(lambda x: int(str(x)[:-4]))
    df['交易月'] = df['交易年月日'].apply(lambda x: int(str(x)[3:5]))

    df.sort_values(by='交易年月日', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)

    return df


def fn_str_2_num(string):
    if type(string) == str:
        str_2_num = ''
        for a in string:
            try:
                a = int(a)
            except:
                pass
            str_2_num += str(a)
        return str_2_num
    else:
        print(f'Invalid type ! expect str but input {type(string)}')
        return string


def fn_addr_handle(addr):
    addr = fn_str_2_num(addr)

    a = addr.split('號')[0] + '號' if '號' in addr else addr
    a = a.split('棟')[0][0:-1] if '棟' in a else a

    for C in ['Ａ', 'Ｂ', 'Ｃ', 'D', 'Ｆ']:
        a = a.split(C)[0] if C in a else a

    A = re.findall(r'[A-Z][a-z]*', a)
    a = a.split(A[0])[0] if len(A) else a
    a = a[0:-1] if a.endswith('旁') else a

    a = a.replace(' ', '')
    a = a.replace('公?路', '公館路') if '北投區公?路' in a else a
    a = a.replace('公路', '公館路') if '北投區公路' in a else a
    a = a.replace('公 路', '公館路') if '北投區公 路' in a else a
    a = a.replace('臺北市', '台北市')

    a = '台北市南港區興南街98號' if a == '台北市南港區98號' else a
    a = '台北市北投區公館路243號' if a == '台北市北投區公?路' else a

    a = '台北市北投區大度路三段301巷67號' if a == '台北市北投區關渡里大度路三段301巷' else a  # 康寶日出印象
    a = '台北市北投區大度路三段301巷67號' if a == '台北市北投區大度路三段301巷' else a  # 康寶日出印象
    a = '台北市北投區文林北路175號' if a == '台北市北投區文林北路175巷及75巷口' else a  # 華固文臨
    a = '台北市北投區西安街二段197號' if a == '台北市北投區西安街二段吉利街口' else a  # 華固翡儷

    return a


@fn_profiler
def fn_house_coor_read():
    House_coor_file = os.path.join(dic_of_path['database'], 'House_coor.csv')
    if os.path.exists(House_coor_file):
        df_coor_read = pd.read_csv(House_coor_file, encoding='utf_8_sig', index_col=0)
        df_coor_read.index = df_coor_read.index.map(fn_str_2_num)
        df_coor_read = df_coor_read.reset_index().drop_duplicates(subset='index', keep='first')
        df_coor_read.set_index('index', inplace=True)
    else:
        df_coor_read = pd.DataFrame()

    return df_coor_read


@fn_profiler
def fn_house_coor_save(df_coor_save):
    House_coor_file = os.path.join(dic_of_path['database'], 'House_coor.csv')
    df_coor_save = df_coor_save.reset_index().drop_duplicates(subset='index', keep='first')
    df_coor_save.set_index('index', inplace=True)
    df_coor_save.sort_index(inplace=True)
    df_coor_save.to_csv(House_coor_file, encoding='utf_8_sig')
    print(f'儲存 {df_coor_save.shape[0]} 筆座標 至 {House_coor_file}')


def fn_save_building_name(path):
    csv = os.path.join(path, 'output\\house_all.csv')
    if not os.path.exists(csv):
        print(f'{csv} not existed !!!')
        return

    df_csv = pd.read_csv(csv)
    df_coor = fn_house_coor_read()
    df_coor = df_coor.astype(str)

    df_csv = df_csv[~df_csv['建案名稱'].isna()]
    df_csv['土地位置建物門牌'] = df_csv['土地位置建物門牌'].apply(fn_addr_handle)
    df_csv.drop_duplicates(subset=['土地位置建物門牌'], keep='first', inplace=True)
    print(df_csv.shape)

    list_of_addr = df_csv['土地位置建物門牌'].tolist()

    for idx in df_coor.index:
        if idx in list_of_addr:
            where = list_of_addr.index(idx)
            build_case = df_csv[['建案名稱']].iloc[where, :]
            build_name = None if build_case[-1].endswith('區') and len(build_case[-1]) == 3 else build_case[-1]
            df_coor.at[idx, 'Build case'] = build_name

        dic_of_dist = fn_get_admin_dist(idx)
        for k in dic_of_dist.keys():
            try:
                df_coor.at[idx, k] = dic_of_dist[k]
            except:
                assert False, f'{idx}, {k}, {dic_of_dist[k]}'

    fn_house_coor_save(df_coor)


def fn_gen_ave(df):
    df['coor'] = df['log'].astype(str) + '_' + df['lat'].astype(str)
    ave_by = '每坪單價(萬)'
    df_mrt_ave = df.groupby(['MRT'], as_index=True)[ave_by].mean()
    df_sku_ave = df.groupby(['sku_name'], as_index=True)[ave_by].mean()
    df_coor_ave = df.groupby(['coor'], as_index=True)[ave_by].mean()
    df_dist_ave = df.groupby(['鄉鎮市區'], as_index=True)[ave_by].mean()

    path = os.path.join(dic_of_path['database'], 'SKU_ave.csv')
    df_sku_ave = df_sku_ave.round(2)
    df_sku_ave.sort_values(inplace=True)
    df_sku_ave.to_csv(path, encoding='utf-8-sig')

    path = os.path.join(dic_of_path['database'], 'MRT_ave.csv')
    df_mrt_ave = df_mrt_ave.round(2)
    df_mrt_ave.sort_values(inplace=True)
    df_mrt_ave.to_csv(path, encoding='utf-8-sig')

    path = os.path.join(dic_of_path['database'], 'DIST_ave.csv')
    df_dist_ave = df_dist_ave.round(2)
    df_dist_ave.sort_values(inplace=True)
    df_dist_ave.to_csv(path, encoding='utf-8-sig')

    latest_rel = '111_0501'
    for idx in df.index:
        mrt_1 = df.loc[idx, 'MRT']
        df.at[idx, 'MRT_ave'] = round(df_mrt_ave[mrt_1], 2)

        sku = df.loc[idx, 'sku_name']
        df.at[idx, 'SKU_ave'] = round(df_sku_ave[sku], 2)

        coor = df.loc[idx, 'coor']
        df.at[idx, 'coor_ave'] = round(df_coor_ave[coor], 2)

        dist = df.loc[idx, '鄉鎮市區']
        df.at[idx, 'DIST_ave'] = round(df_dist_ave[dist], 2)

        df.at[idx, 'Latest'] = 0 if latest_rel in df.loc[idx, 'File'] else 1

    return df


def fn_get_house_hold(addr):
    A = re.findall(r'[A-Z][a-z]*', addr)
    if len(A):
        house_hold = A[0] + addr.split(A[0])[-1]
        house_hold = house_hold.split('-')[0]
        house_hold = house_hold.split('棟')[0]
        house_hold = house_hold.split('戶')[0]
    else:
        house_hold = 'X'

    return house_hold + '戶' if '戶' not in house_hold else house_hold


def fn_gen_build_case(df):
    # print('fn_gen_build_case start')
    house_typ = '預售屋' if len(df['建築完成年月'].unique()) == 1 else '中古屋'

    this_yr = datetime.datetime.now().year - 1911

    if '屋齡' not in df.columns:
        if house_typ == '中古屋':
            # df['屋齡']=df['建築完成年月'].apply(lambda x:this_yr - int(str(x):-4])  # 1090430
            df['屋齡'] = df['交易年月日'].astype(int) - df['建築完成年月'].astype(int)
            df['屋齡'] = df['屋齡'].apply(lambda x: 0 if int(x) < 10000 else int(str(x)[:-4]))
        else:
            df['屋齡'] = df['建築完成年月'].apply(lambda x: 0)

    if '建案名稱' not in df.columns:
        df['建案名稱'] = df['鄉鎮市區']

    df_coor_read = fn_house_coor_read()
    for idx in df.index:
        if str(df.loc[idx, '建案名稱']) == 'nan' or str(df.loc[idx, '建案名稱']) == str(df.loc[idx, '鄉鎮市區']):
            addr = df.loc[idx, '土地位置建物門牌']
            addr = fn_addr_handle(addr)
            if addr in df_coor_read.index:
                if 'Build case' in df_coor_read.columns:
                    build_case = str(df_coor_read.loc[addr, 'Build case'])
                    if build_case != 'nan' and build_case != 'NA' and not build_case.endswith('區'):
                        df.at[idx, '建案名稱'] = build_case
                        print(addr, '-->', df.at[idx, '建案名稱'])

    # print('fn_gen_build_case end')
    return df


def fn_gen_house_data(file, post, slp=5, df_validate=pd.DataFrame(), is_trc=True):
    df = pd.read_csv(file) if df_validate.shape[0] == 0 else df_validate
    source = os.path.dirname(file)
    root = os.path.dirname(source)
    output = os.path.join(root, 'output')
    if not os.path.exists(output):
        print(f'{output} Not existed, Create it I')
        os.makedirs(output)

    output = os.path.join(output, 'house_' + post + '.csv')

    df['File'] = df['土地位置建物門牌'].apply(lambda x: file.split('\\')[-1])
    df = fn_data_cleaning(df)

    if df.shape == (0, 0):
        return df

    list_of_addr = df['土地位置建物門牌'].unique()

    list_of_addr_unique = []
    for addr in list_of_addr:
        a = fn_addr_handle(addr)

        if a not in list_of_addr_unique:
            list_of_addr_unique.append(a)

    dic_of_geo_info = dict()
    dic_of_coor = {}
    df_coor_read = fn_house_coor_read()

    coor_save = []
    for addr in list_of_addr_unique:
        # if addr not in df_coor_read.index:
        try:
            dic_of_geo_info[addr], is_coor_save, is_match, addr_fr_db = fn_get_geo_info(addr, df_coor_read, slp)
            coor_save.append(is_coor_save)
        except:
            print(addr, addr in df_coor_read.index)
            # print(df_coor_read.loc[addr, :])
            assert False, f'fn_get_geo_info Fail: {addr} {addr in df_coor_read.index}'

        # if addr not in df_coor_read.index:
        dic_of_coor[addr] = [dic_of_geo_info[addr]['coor']['lat'], dic_of_geo_info[addr]['coor']['log']]
        if is_trc:
            print(f'{is_coor_save}, {list_of_addr_unique.index(addr)}/{len(list_of_addr_unique)}', addr,
                  dic_of_coor[addr])

    if len(dic_of_coor.keys()):
        df_coor = pd.DataFrame(dic_of_coor, index=['lat', 'lon'])
        df_coor = df_coor.T

        for idx in df_coor.index:
            dic_of_dist = fn_get_admin_dist(idx)
            for k in dic_of_dist.keys():
                df_coor.at[idx, k] = dic_of_dist[k]

        assert len(coor_save) == df_coor.shape[0], f'coor_save {len(coor_save)} df_coor rows {df_coor.shape[0]}'
        drops = []
        for i in range(len(coor_save)):
            if coor_save[i] is False:
                drops.append(i)
                # print(f'drop {df_coor.index[i]} since coor_save[{i}]={coor_save[i]}')

        if len(drops):
            drop_idx = []
            for d in drops:
                drop_idx.append(df_coor.index[d])
                # print(f'drop df_coor.index[{d}] = {df_coor.index[d]}')
            df_coor.drop(index=drop_idx, inplace=True)

        if df_coor.shape[0] > 0:
            print(f'新增了 {df_coor.shape[0]} 筆座標 !')
            # df_coor_save = df_coor_read.append(df_coor)
            df_coor_save = pd.concat([df_coor_read, df_coor])
            fn_house_coor_save(df_coor_save)

    for idx in df.index:
        addr = df['土地位置建物門牌'][idx]
        a = fn_addr_handle(addr)

        if a in dic_of_geo_info.keys():
            for dic in dic_of_geo_info[a].keys():
                for k in dic_of_geo_info[a][dic].keys():
                    df.at[idx, k] = dic_of_geo_info[a][dic][k]
        else:
            print(f'{a} not in dic_of_geo_info.keys()')

    # print(df.columns)
    df.to_csv(output, encoding='utf_8_sig', index=False)
    return df


@fn_profiler
def fn_gen_raw_data(path, slp=5, is_force=True):
    path_source = os.path.join(path, 'source')
    path_output = os.path.join(path, 'output/house_all.csv')

    if 'pre_owned_house' in path:
        key = '_lvr_land_a'
    else:
        key = '_lvr_land_b'

    file_2_read = []
    for i, j, files in os.walk(path_source):
        for f in files:
            print(f)
            if key in f.lower() and '.csv.bak' not in f:
                post = f.lower().split(key)[-1].split('.csv')[0]
                post = f.lower().split(key)[0] + post
                file_2_read.append((f, post))

    if len(file_2_read) == 0:
        pprint.pprint(file_2_read)
        print(path_source)

    df_all = pd.DataFrame()
    if is_force is False and os.path.exists(path_output):
        df_all = pd.read_csv(path_output)
        df_all.drop(columns=['屋齡'], inplace=True)

    for f in file_2_read:
        info = f'({file_2_read.index(f) + 1}/{len(file_2_read)})'
        dir_f = os.path.join(path_source, f[0])
        is_new = False
        if 'File' in df_all.columns:
            is_new = f[0] not in df_all['File'].values
        if is_force or is_new:  # or '111_1201' in f[0]:
            print(info, 'Parsing:', f[0])
            df = fn_gen_house_data(dir_f, f[1], slp)
            # df_all = df_all.append(df, ignore_index=True)
            if len(df.columns) != len(df_all.columns):
                print(f'{df_all.shape}, {df_all.columns}')
                print(f'{df.shape}, {df.columns}')
            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(info, 'Existed: ', f[0])

    if df_all.shape[0]:
        print(f'共{df_all.shape[0]}筆交易資料')
        df_all = fn_gen_build_case(df_all)
        df_all.sort_values(by='交易年月日', inplace=True, ascending=False)
        df_all = fn_gen_ave(df_all.copy())
        df_all.to_csv(path_output, encoding='utf_8_sig', index=False)
        print(df_all['備註'].value_counts())


def fn_gen_vill(file):
    shapes, properties = fn_read_shp()

    if '_coor.csv' in file:
        df_all = pd.read_csv(file, index_col=0)
        df_all = df_all.reset_index().drop_duplicates(subset='index', keep='first').set_index('index').sort_index()
    else:
        df_all = pd.read_csv(file)

    # add vill from coor
    is_update = False

    for i in df_all.index:
        if '里' in df_all.columns and str(df_all.loc[i, '里']).endswith('里'):
            pass
        else:
            lon = df_all.loc[i, 'log'] if 'log' in df_all.columns else df_all.loc[i, 'lon']
            lat = df_all.loc[i, 'lat'] if 'lat' in df_all.columns else df_all.loc[i, '緯度']

            try:
                vill_info = fn_search_vill(lon, lat, shapes, properties)
            except:
                assert False, f'lon: {lon} lat: {lat}, {i}'

            if vill_info != 'Unknown':
                # city = vill_info.split(',')[0].replace(' ', '')
                dist = vill_info.split(',')[1].replace(' ', '')
                vill = vill_info.split(',')[2].replace(' ', '')

            col = '鄉鎮市區' if '鄉鎮市區' in df_all.columns else '區'
            if vill_info != 'Unknown' and df_all.loc[i, col] == dist:
                df_all.at[i, '里'] = vill
                is_update = True
                # print(f'{i}/{df_all.shape[0]}', vill, dist, df_all.loc[i, '地址'])
            else:
                addr = df_all.loc[i, "地址"] if "地址" in df_all.columns else i
                print(f'Error {i}/{df_all.shape[0]}, {vill}, {dist}, ({lon}, {lat}), {df_all.loc[i, col]}, {addr}')

    if is_update:
        if '_coor.csv' in file:
            df_all.to_csv(file, encoding='utf_8_sig')
        else:
            df_all.to_csv(file, encoding='utf_8_sig', index=False)

        print(f'Vill updated !')

# should import from house_utils.py
# dic_bc_rename = {
#     '中星仁愛旭': '仁愛旭',
#     '宏國大道城A棟': '宏國大道城-A區',
#     '宏國大道城B棟': '宏國大道城-B區',
#     '宏國大道城C棟': '宏國大道城-C區',
#     '宏國大道城D棟': '宏國大道城-D區',
#     '政大爵鼎NO2': '政大爵鼎NO.2',
#     '政大爵鼎NO1': '政大爵鼎NO.1',
#     '吉美君悅': '吉美君悦',
#     '忠泰衍見築': '衍見築',
#     '台大學': '台太學',
#     '寶舖ＣＡＲＥ': '寶舖CARE',
#     '三磐舍紫II': '三磐舍紫2',
#     '德杰羽森-璽': '德杰羽森',
#     '德杰羽森-琚': '德杰羽森',
#     '德杰羽森-玥': '德杰羽森',
#     '吉吉美': '喆美',
#     '韋昌?青江': '韋昌画清江',
#     '睿泰川? ': '睿泰川矅',
# }


def fn_gen_bc_info():
    file_df_all = os.path.join(dic_of_path['root'], 'pre_sold_house', 'output', 'house_all.csv')
    df = pd.read_csv(file_df_all, encoding='utf-8-sig')

    file_df_bc = os.path.join(dic_of_path['database'], 'build_case_info_ext.csv')
    df_bc_info = pd.read_csv(file_df_bc, encoding='utf-8-sig')

    file_df_lg = os.path.join(dic_of_path['database'], 'builder_litigation.csv')
    df_lg_info = pd.read_csv(file_df_lg, encoding='utf-8-sig')

    '''
    行政區	
    建照年度	
    完工年度	
    公開銷售	
    建案名稱	
    基地面積(坪)	
    建蔽面積(坪)	
    建蔽率(%)	
    容積率(%)	
    公設比(%)	
    棟數	地上樓層	
    地下樓層	總戶數	平面車位	機械車位	總車位數	預估工期	建造執照	預估工期	投資建設	建築設計	營造公司	企劃銷售	結構工程	座向規劃	車位規劃	車位配比	用途規劃	土地分區	管理費用	景觀設計	公設設計	燈光設計	棟戶規劃	樓層規劃	基地地址	建材說明	url	爬蟲耗時(秒)	更新日期		
    '''

    bc_infos_all = ['投資建設', '營造公司', '建造執照', '企劃銷售', '公開銷售', '預估工期', '座向規劃', '土地分區',
                    '完工年度', '建蔽率(%)', '基地面積(坪)', '建蔽面積(坪)', '容積率(%)', '公設比(%)',
                    '地上樓層', '地下樓層', '總戶數', '平面車位', '機械車位']

    bc_infos = [i for i in bc_infos_all if i in df_bc_info.columns]

    cols = bc_infos + list(df.columns)
    bc_lack = []
    if '建案名稱' in df.columns:
        for idx in df.index:
            bc = df.loc[idx, '建案名稱']
            bc = dic_bc_rename[bc] if bc in dic_bc_rename.keys() else bc
            if str(bc) != 'nan':
                if bc in df_bc_info['建案名稱'].values:
                    df_sel_bc = df_bc_info[df_bc_info['建案名稱'] == bc]
                    for info in bc_infos:
                        assert info in df_sel_bc.columns, f'{info} not in df_sel_bc.columns'
                        df.at[idx, info] = df_sel_bc[info].values[0]
                else:
                    if bc not in bc_lack:
                        print(f'build_case_info_ext.csv has no {bc}')
                        bc_lack.append(bc)
    else:
        assert False, f'建案名稱 NOT in df.columns'

    for idx in df.index:
        builder = df.loc[idx, '投資建設']
        constructor = df.loc[idx, '營造公司']
        lg_num_b = df_lg_info[df_lg_info['建商營造'] == builder]['歷年案件'].values[0] if builder in df_lg_info['建商營造'].values else 0
        lg_num_c = df_lg_info[df_lg_info['建商營造'] == constructor]['歷年案件'].values[0] if constructor in df_lg_info['建商營造'].values else 0
        df.at[idx, '建商訴訟'] = lg_num_b if lg_num_b < 2500 else 0
        df.at[idx, '營造訴訟'] = lg_num_c if lg_num_c < 2500 else 0
        df.at[idx, '所有訴訟'] = df.at[idx, '建商訴訟'] + df.at[idx, '營造訴訟']

    df.to_csv(file_df_all, encoding='utf-8-sig', index=False)


def fn_gen_tax_info(file):
    tax = os.path.join(dic_of_path['database'], '108_165-A.csv')  # 107_165-A.csv, 108_165-A.csv
    df_tax = pd.read_csv(tax)
    df_all = pd.read_csv(file)

    if '里' in df_all.columns:
        df_all['區_里'] = df_all['鄉鎮市區'] + '_' + df_all['里']
        if '鄉鎮市區' in df_tax.columns:
            df_tax['區_里'] = df_tax['鄉鎮市區'] + '_' + df_tax['村里']
        else:
            df_tax['區_里'] = df_tax['行政區'] + '_' + df_tax['里']

        for idx in df_all.index:
            d_v = df_all.loc[idx, '區_里']
            if d_v in df_tax['區_里'].values:
                df_tax_sel = df_tax[df_tax['區_里'] == d_v]
                assert df_tax_sel.shape[0] == 1, f'df_tax_sel.shape = {df_tax_sel.shape} {d_v} {idx}'
                for c in df_tax_sel.columns:
                    if c in ['里', '行政區', '鄉鎮市區', '村里', '區_里', '納稅單位']:
                        pass
                    else:
                        df_all.at[idx, '稅_' + str(c)] = df_tax_sel[c].values[0]
            else:
                if str(d_v) != 'nan':
                    print(f'{d_v} not in df_tax {tax}')

        df_all.to_csv(file, encoding='utf-8-sig', index=False)
        print(f'Add Tax Info Done ! {df_all.shape} {file}')
    else:
        print(f'Tax gen fail ! Missing column 里 in {file}')


def fn_main():
    # XX path = os.path.join(dic_of_path['root'], 'pre_owned_house')

    path = os.path.join(dic_of_path['root'], 'pre_sold_house')
    fn_gen_raw_data(path, slp=3, is_force=False)

    # XX fn_save_building_name(path)

    file = os.path.join(path, 'output', 'house_all.csv')
    fn_gen_vill(file)
    fn_gen_tax_info(file)

    file = os.path.join(dic_of_path['database'], 'House_coor.csv')
    fn_gen_vill(file)

    fn_gen_bc_info()


if __name__ == '__main__':
    ts = time.time()

    fn_main()

    te = time.time()
    dur = te - ts

    print(f'execution time: {int(dur / 60)} 分 {int(dur % 60)} 秒')
