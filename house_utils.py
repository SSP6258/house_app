import time
import datetime
import os
import numpy as np
import pandas as pd
import cn2an
import random
import geopandas as gpd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from geopy.distance import geodesic
from collections import defaultdict
from workalendar.asia import Taiwan
from shapely.geometry import shape, Point
# from tabula import read_pdf

dic_of_path = {
    # 'root': r'D:\05_Database\house_data',
    # 'database': r'D:\05_Database\house_data\database'
    'root': 'house_data',
    'database': 'house_data/database',
}

dic_of_bc_info = defaultdict(list)

dic_of_capacity_ratio = {
    'Source': '臺北市土地使用分區管制規則有關建蔽率容積率庭院等規定一覽表',
    'url': r'https://www.udd.gov.taipei/laws/rdpqpr5-5426',
    '更新日期': '110/12/08',
    '承辦人': '[都市規劃科] 關仲芸',
    '聯絡電話': '02-27208889 轉8265',
    '住ㄧ': 60,
    '住二': 120,
    '住二之一': 160,
    '住二之二': 225,
    '住三': 225,
    '住三之一': 300,
    '住三之二': 400,
    '住四': 300,
    '住四之一': 400,
    '商ㄧ': 360,
    '商二': 630,
    '商三': 560,
    '商四': 800,
    '工二': 200,
    '工三': 300,
}


def fn_profiler(func):
    def wrapper(*args, **kwargs):
        ts = datetime.datetime.now()
        val = func(*args, **kwargs)
        te = datetime.datetime.now()
        dur = te - ts
        dur_us = dur.microseconds
        if dur_us > 1e6:
            excu_time = f'{int(dur_us / 1e6)} (sec)'
        elif dur_us >= 1e3:
            excu_time = f'{int(dur_us / 1e3)} (ms)'
        else:
            excu_time = f'{dur_us} 微秒(us)'

        if dur_us > 500 * 1000:
            print(func.__name__, excu_time)

        return val

    return wrapper


def fn_get_coordinate(addr, slp=5):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    chromedriver = os.path.join(dic_of_path['database'], 'chromedriver.exe')
    browser = webdriver.Chrome(executable_path=chromedriver, options=options)
    browser.get("http://www.map.com.tw/")

    # search = browser.find_element_by_id("searchword")
    search = browser.find_element(by=By.ID, value="searchWord")
    search.clear()
    time.sleep(random.randint(1, 3))
    search.send_keys(addr)
    browser.find_element(by=By.XPATH, value="/html/body/form/div[10]/div[2]/img[2]").click()
    slp = slp + random.randint(5, 15)
    time.sleep(slp)
    iframe = browser.find_elements(by=By.CLASS_NAME, value="winfoIframe")[0]

    browser.switch_to.frame(iframe)
    coor_btn = browser.find_element(by=By.XPATH,
                                    value="/html/body/form/div[4]/table/tbody/tr[3]/td/table/tbody/tr/td[2]")

    coor_btn.click()
    time.sleep(random.randint(1, 3))
    coor = browser.find_element(by=By.XPATH, value="/html/body/form/div[5]/table/tbody/tr[2]/td")
    coor = coor.text.strip().split(" ")

    lat = coor[-1].split('：')[-1]
    lon = coor[0].split('：')[-1]
    # print(coor, coor[0], coor[-1], lat, lon)

    browser.quit()

    return (lat, lon)


def fn_gen_mrt_coor():
    csv_path = os.path.join(dic_of_path['database', 'MRT_coor.csv'])
    df = pd.read_csv(csv_path)
    for col in df.columns:
        df = df.drop(columns=col) if 'Unnamed' in col else df
    is_save = False
    print(df.columns)
    for idx in df.index:
        mrt = df['Station'][idx]
        mrt_addr = df['Addr'][idx]
        mrt_dist = mrt_addr.split('市')[-1].split('區')[0] + '區'
        df['District'] = df['District'].astype(type(mrt_dist))

        if df.loc[idx, 'District'] == 'nan':
            df.at[idx, 'District'] = mrt_dist
            is_save = True
        if np.isnan(df.loc[idx, 'lat']):
            mrt_coor = fn_get_coordinate(mrt_addr)
            df.at[idx, 'lat'] = mrt_coor[0]
            df.at[idx, 'log'] = mrt_coor[1]
            is_save = True
            print(idx, mrt, mrt_addr, mrt_coor)

    if is_save:
        print('update mrt_coor.csv !')
        df.to_csv(csv_path, encoding='utf_8_sig', index=False)


def fn_get_travel_time():
    path = dic_of_path['database']
    file = os.path.join(path, 'MRT_travel_time.csv')
    # options = webdriver.ChromeOptions()
    # options.add_argument("headless")
    # driver = webdriver.Chrome(options=options)
    driver = webdriver.Chrome()
    driver.implicitly_wait(3)

    dic_mrt = {}
    df_s = pd.DataFrame()

    NO = [range(0, 7), range(181, 201), range(135, 174)]
    NO += [range(112, 121), range(106, 109), range(72, 76)]

    for i in range(214):

        bypass = False
        for no in NO:
            if i in no:
                bypass = True
                break

        if bypass:
            continue

        if len(str(i)) == 1:
            mrt = '00' + str(i)
        elif len(str(i)) == 2:
            mrt = '0' + str(i)
        else:
            mrt = str(i)

        link = f'https://web.metro.taipei/pages/tw/ticketroutetimesingle/{mrt}'
        xpath = "/html/body/app-root/app-ticketroutetimesingle/section[2]/table"

        time.sleep(2)
        driver.get(link)

        try:
            df = pd.read_html(driver.find_element(By.XPATH, xpath).get_attribute('outerHTML'))[0]
        except:
            print(f'No {mrt}')
            df = pd.DataFrame()

        if df.shape[0]:
            fr = df.iloc[0, 0]
            is_cross = len(fr.split(' ')) == 3
            print(mrt, '-->', is_cross, fr, df.shape[0])
            df = df[df.columns[[2, -1]]]
            dic_mrt['to'] = df[df.columns[0]].tolist()
            dic_mrt[f'to_{mrt}_time'] = df[df.columns[1]].tolist()

            dic_mrt['to'].append(fr)
            dic_mrt[f'to_{mrt}_time'].append(0)
            if is_cross:
                dic_mrt['to'].append(fr.split(' ')[-2] + fr.split(' ')[-1])
                dic_mrt[f'to_{mrt}_time'].append(0)

            df_temp = pd.DataFrame(dic_mrt)
            df_temp.sort_values(by='to', inplace=True)
            df_temp.set_index('to', inplace=True)
            df_s[fr] = df_temp[f'to_{mrt}_time']
        else:
            print(f'No {mrt}')

    driver.quit()
    if df_s.shape[0]:
        df_s.fillna(0, inplace=True)
        df_s.to_csv(file, encoding='utf-8-sig')


def fn_is_working_day(x):
    day = datetime.date.fromisoformat(x)
    is_working_day = Taiwan().is_working_day(day)

    return is_working_day


def fn_cn_2_an(string):
    s_cn2an = []
    for i in range(len(string)):
        try:
            s_cn2an.append(cn2an.cn2an(string[i]))
        except:
            s_cn2an.append(string[i])

    string_an = ''.join(str(e) for e in s_cn2an)

    return string_an


def fn_get_admin_dist(addr, is_trc=True):
    city = addr.split('市')[0] + '市'
    dist = addr.split(city)[-1].split('區')[0] + '區'
    vil = addr.split(dist)[-1].split('里')[0] + '里' if '里' in addr and '八里區' not in dist else 'NA'

    nb = addr.split('鄰')[0] + '鄰' if '鄰' in addr else 'NA'
    if nb != 'NA':
        nb = nb.split(vil)[-1] if vil in nb else nb
        nb = nb.split(dist)[-1] if dist in nb else nb

    road = addr.split('路')[0] + '路' if '路' in addr else 'NA'
    if road != 'NA':
        road = road.split(nb)[-1] if nb in road else road
        road = road.split(vil)[-1] if vil in road else road
        road = road.split(dist)[-1] if dist in road else road

    street = addr.split('街')[0] + '街' if '街' in addr else 'NA'
    if street != 'NA':
        street = street.split(nb)[-1] if nb in street else street
        street = street.split(vil)[-1] if vil in street else street
        street = street.split(dist)[-1] if dist in street else street
        street = street.split(road)[-1] if road in street else street

    section = addr.split('段')[0] + '段' if '段' in addr else 'NA'
    if section != 'NA':
        section = section.split(nb)[-1] if nb in section else section
        section = section.split(vil)[-1] if vil in section else section
        section = section.split(dist)[-1] if dist in section else section
        section = section.split(road)[-1] if road in section else section
        section = section.split(street)[-1] if street in section else section

    lane = addr.split('巷')[0] + '巷' if '巷' in addr else 'NA'
    if lane != 'NA':
        lane = lane.split(nb)[-1] if nb in lane else lane
        lane = lane.split(vil)[-1] if vil in lane else lane
        lane = lane.split(dist)[-1] if dist in lane else lane
        lane = lane.split(road)[-1] if road in lane else lane
        lane = lane.split(street)[-1] if street in lane else lane
        lane = lane.split(section)[-1] if section in lane else lane

    lane2 = addr.split('弄')[0] + '弄' if '弄' in addr else 'NA'
    if lane2 != 'NA':
        lane2 = lane2.split(nb)[-1] if nb in lane2 else lane2
        lane2 = lane2.split(vil)[-1] if vil in lane2 else lane2
        lane2 = lane2.split(dist)[-1] if dist in lane2 else lane2
        lane2 = lane2.split(road)[-1] if road in lane2 else lane2
        lane2 = lane2.split(street)[-1] if street in lane2 else lane2
        lane2 = lane2.split(section)[-1] if section in lane2 else lane2
        lane2 = lane2.split(lane)[-1] if lane in lane2 else lane2

    num = addr.split('號')[0] + '號' if '號' in addr else 'NA'
    if num != 'NA':
        num = num.split(nb)[-1] if nb in num else num
        num = num.split(vil)[-1] if vil in num else num
        num = num.split(dist)[-1] if dist in num else num
        num = num.split(road)[-1] if road in num else num
        num = num.split(street)[-1] if street in num else num
        num = num.split(section)[-1] if section in num else num
        num = num.split(lane)[-1] if lane in num else num
        num = num.split(lane2)[-1] if lane2 in num else num

        if num.split('號')[0].isnumeric():
            pass
        else:
            # 7.9號 / 39，41號 / 8，10號 / 四小段11號 / 二小段0618-0000號
            number = num.split('號')[0].split('之')[0].split('-')[0].split('~')[0].split('.')[0].split('，')[0].split('段')[
                -1]
            if number.isnumeric():
                num = number + '號'
            else:
                n = ''
                for i, v in enumerate(num.split('號')[0]):
                    n += v if v.isnumeric() else ''
                if is_trc:
                    print('Review this addr num !', num, '-->', n + '號')
                num = n + '號'

    if section != 'NA':
        section = fn_cn_2_an(section)

    # special addr handle
    if road == 'NA':
        if '大道' in num:
            road = num.split('道')[0] + '道'
            num = num.split('道')[-1]

        if '大道' in section:
            road = section.split('道')[0] + '道'
            section = section.split('道')[-1]

        if '中正東硌' in section:
            road = section.split('硌')[0] + '路'
            section = section.split('硌')[-1]

    dic_of_dist = {
        '市': city,
        '區': dist,
        '里': vil,
        '鄰': nb,
        '路': road,
        '街': street,
        '段': section,
        '巷': lane,
        '弄': lane2,
        '號': num,
    }

    # print(addr)
    # print(dic_of_dist)

    return dic_of_dist


def fn_get_coor_fr_db(addr, df_coor, is_trc=True):
    dic_of_dist = fn_get_admin_dist(addr, is_trc)

    for k, v in dic_of_dist.items():
        if v != 'NA' and k in df_coor.columns and v in df_coor[k].values:
            df_coor = df_coor[df_coor[k] == v]
        # print(k, v, v in df_coor[k].values, df_coor.shape[0])

    sel = 0
    matched = False
    if df_coor.shape[0]:
        for a in ['號', '弄', '巷']:
            if a in dic_of_dist.keys() and a in dic_of_dist[a] and not matched:
                num = int(dic_of_dist[a].split(a)[0])
                df_coor_sel = df_coor[df_coor[a].apply(lambda x: str(x) != 'nan' and str(x) != 'NA')]

                try:
                    nums = df_coor_sel[a].apply(
                        lambda x: x if str(x) == 'nan' else int(str(x)
                                                                .split(a)[0]
                                                                .split('之')[0]
                                                                .split('-')[0]
                                                                .split('，')[0])).tolist()
                except:
                    assert False, f'{a}, {df_coor[a]}'

                diff = [abs(n - num) for n in nums if str(n) != 'nan']
                if len(diff):
                    sel = diff.index(min(diff))
                    matched = True
                    df_coor = df_coor_sel
                    if is_trc:
                        print(a, num, nums, sel, nums[sel], matched, len(diff))
                    break

    coor = df_coor[['lat', 'lon']].iloc[sel, :]
    coor = tuple(coor)
    addr_match = df_coor.index[sel]

    if is_trc:
        if matched:
            print('Coor From DB: ', addr, ' --> ', addr_match, coor)
        else:
            print(f'can NOT find similar addr of {addr}')

    return coor, matched, addr_match


def fn_get_mrt_travel_time(src, io, df_mrt, df_time):
    ave_travel_time = 0

    if io == '進站':
        remove_src = ['BL', 'Y']
        for r in remove_src:
            src = src.replace(r, '') if r in src else src
        if '橋頭站' in src:
            src = src.replace('站', '')
        if src in df_time.columns:
            df_time_col = df_time[[src]]
        else:
            print('No MRT', src, df_time.columns)
        times = []
        remove_des = ['BL', 'Y']
        for idx in df_mrt.index:
            des = df_mrt.loc[idx, '出站']

            for r in remove_des:
                des = des.replace(r, '') if r in des else des
            if '橋頭站' in des:
                des = des.replace('站', '')

            if des in df_time_col.index:
                t = int(df_time_col.loc[des, src])
                count = int(df_mrt.loc[idx, '人次'])
                times.append(t * count)
            else:
                print('Missing', src, des, df_time_col.index)

        total_count = df_mrt['人次'].sum()
        try:
            ave_travel_time = round(sum(times) / total_count, 2)
        except:
            print('Ave', src, len(times), sum(times), total_count, ave_travel_time)
            assert False

    return ave_travel_time


def fn_get_mrt_tput(io, fr, to):
    path = dic_of_path['database']
    df_time = pd.read_csv(os.path.join(path, 'MRT_travel_time.csv'),
                          index_col=0,
                          encoding='utf-8-sig')
    df_time.rename(index={i: i.split(' ')[-1] for i in df_time.index},
                   columns={i: i.split(' ')[-1] for i in df_time.columns},
                   inplace=True)

    df_time = df_time.loc[~df_time.index.duplicated(), ~df_time.columns.duplicated()]

    if df_time.shape[1] != len(set(df_time.columns)):
        print('duplicate cols !', df_time.columns)

    dic_of_summary = dict()

    for i in range(12):
        m = i + 1
        month = f'0{m}' if m < 10 else m
        file = f'臺北捷運每日分時各站OD流量統計資料_2021{month}.csv'
        if not os.path.exists(os.path.join(path, file)):
            print(f'{file} NOT existed !')
            continue

        date = file.split('_')[-1].split('.csv')[0]  # 202101 ~202109
        month = date[5:] if date[4] == '0' else date[4:]
        year = date[:4]
        try:
            df = pd.read_csv(os.path.join(path, file), encoding='utf-8')
        except:
            df = pd.read_csv(os.path.join(path, file), encoding='utf-16')

        df = df[df['人次'] > 0]
        days = df['日期'].unique()
        dic_of_workday = dict()
        print(file, len(days), ' dates')
        for d in days:
            day = datetime.date.fromisoformat(d)
            dic_of_workday[d] = Taiwan().is_working_day(day)

        # df wd = df[df['日期'].apply(fr_is_working_day)]
        df_wd = df[df['日期'].apply(lambda x: dic_of_workday[x])]

        df_io = df_wd[df_wd['時段'].apply(lambda x: fr <= int(x) <= to)]
        days = len(df_io['日期'].unique())

        dic_of_count = defaultdict(list)
        MRTs = list(df_io[io].unique())
        for mrt in MRTs:
            df_mrt = df_io[df_io[io] == mrt]
            df_mrt.reset_index(inplace=True)
            travel_time = fn_get_mrt_travel_time(mrt, io, df_mrt.copy(), df_time)
            total = df_mrt['人次'].sum()
            daily = int(round(total / days, 0))
            dic_of_count['MRT'].append(mrt)
            dic_of_count[f'{month}月_每日上班{io}人數'].append(daily)
            dic_of_count['MRT_Time'].append(travel_time)

        dic_of_summary[f'{month}月_每日上班人數'] = dic_of_count[f'{month}月_每日上班{io}人數']
        dic_of_summary[f'{month}月平均通勤時間'] = dic_of_count['MRT_Time']

    df = pd.DataFrame(dic_of_summary, index=dic_of_count['MRT'])
    df[f'{io}平均'] = df.mean(axis=1).apply(lambda x: int(x))

    sel_cols = []
    for col in df.columns:
        if '通勤' in col:
            sel_cols.append(col)

    df['平均通勤時間'] = df[sel_cols].mean(axis=1).apply(lambda x: round(x, 2))

    df = df.sort_values(by=f'{io}平均', ascending=False)
    df.to_csv(os.path.join(path, f'MRT_上班時段_{io}人數統計_{year}.csv'), encoding='utf-8-sig')


def fn_get_geo_info(addr, df_addr_coor=pd.DataFrame(), slp=5):
    path = dic_of_path['database']
    coor = os.path.join(path, 'MRT_coor.csv')
    school = os.path.join(path, 'School_info.csv')
    mrt_tput_ul = os.path.join(path, 'MRT_上班時段_進站人數統計_2021.csv')
    mrt_tput_dl = os.path.join(path, 'MRT_上班時段_出站人數統計_2021.csv')
    is_save = True

    df_mrt = pd.read_csv(coor, encoding='utf_8_sig')
    df_mrt_ul = pd.read_csv(mrt_tput_ul, encoding='utf_8_sig', index_col=0)
    df_mrt_dl = pd.read_csv(mrt_tput_dl, encoding='utf_8_sig', index_col=0)
    df_sku = pd.read_csv(school, encoding='utf_8_sig')
    city = addr.split('市')[0] + '市'
    if city in df_sku['city'].unique():
        df_sku = df_sku[df_sku['city'] == city]
        df_sku.reset_index(inplace=True)
    else:
        print(addr, city, df_sku['city'].unique())
        assert False, f'{addr}, {city}, {df_sku["city"].unique()}'

    is_match = False
    add_fr_db = ''
    if addr in df_addr_coor.index:
        lat = round(df_addr_coor.loc[addr]['lat'], 5)
        lon = round(df_addr_coor.loc[addr]['lon'], 5)
        addr_coor = (lat, lon)
        is_save = False  # ?
        # print(addr, '--> known coor: ', addr_coor)
    else:
        try:
            addr_coor = fn_get_coordinate(addr, slp)
            # print(addr, addr_coor)
        except:

            # addr_coor = fn_get_coordinate(addr, slp)

            chromedriver = os.path.join(dic_of_path['database'], 'chromedriver.exe')
            print(f'find {addr} from database, is {chromedriver} existed = {os.path.exists(chromedriver)}')
            addr_coor, is_match, add_fr_db = fn_get_coor_fr_db(addr, df_addr_coor.copy())
            # addr_coor = [0, 0]
            is_save = False

    coor_info = dict()
    coor_info['lat'] = addr_coor[0]
    coor_info['log'] = addr_coor[1]

    list_of_mrt_dist = []
    list_of_sku_dist = []

    for idx in df_mrt.index:
        mrt_coor = df_mrt[['lat', 'log']].values[idx]
        # print(f'{addr_coor}, {mrt_coor}')
        d = int(geodesic(addr_coor, mrt_coor).meters)
        list_of_mrt_dist.append(d)

    for idx in df_sku.index:
        sku_coor = df_sku[['lat', 'lon']].values[idx]
        # print(idx, addr_coor, sku_coor)
        d = int(geodesic(addr_coor, sku_coor).meters)
        list_of_sku_dist.append(d)

    min_dist = min(list_of_mrt_dist)
    min_idx = list_of_mrt_dist.index(min_dist)
    mrt_s = df_mrt['Station'][min_idx]
    mrt_l = df_mrt['Line'][min_idx]
    mrt = mrt_s.replace('站', '') if mrt_s != '台北車站' else mrt_s
    mrt_info = dict()
    mrt_info['MRT'] = mrt_l + '線_' + mrt_s
    mrt_info['MRT_DIST'] = min_dist

    mrt = mrt + '站' if mrt == '大橋頭' else mrt

    if mrt in df_mrt_ul.index:
        # print(df_mrt_ul.keys())
        mrt_info['MRT_Tput_UL'] = df_mrt_ul.loc[mrt, '進站平均']
        mrt_info['MRT_Commute_Time_UL'] = df_mrt_ul.loc[mrt, '平均通勤時間']
    else:
        print(f'{mrt} not in df_mrt_ul {df_mrt_ul.index}')
        mrt_info['MRT_Tput_UL'] = 0

    if mrt in df_mrt_dl.index:
        mrt_info['MRT_Tput_DL'] = df_mrt_dl.loc[mrt, '出站平均']
    else:
        print(f'{mrt} not in df_mrt_dl {df_mrt_dl.index}')
        mrt_info['MRT_Tput_DL'] = 0

    mrt_info['MRT_Tput'] = mrt_info['MRT_Tput_DL'] - mrt_info['MRT_Tput_UL']

    min_dist = min(list_of_sku_dist)
    min_idx = list_of_sku_dist.index(min_dist)
    sku_info = dict()
    sku_info['sku_dist'] = min_dist
    sku_info['sku_name'] = df_sku['schoolname'][min_idx]
    sku_info['sku_109_total'] = df_sku['109_Total'][min_idx]
    sku_info['sku_public'] = df_sku['public'][min_idx]

    for k, v in sku_info.items():
        assert v is not None, f'{addr, k, v, min_idx}'

    geo_info = dict()
    geo_info['mrt'] = mrt_info
    geo_info['sku'] = sku_info
    geo_info['coor'] = coor_info

    return geo_info, is_save, is_match, add_fr_db


def fn_gen_school_peoples(path, years):
    # 縣市代碼, 縣市名稱, 鄉鎮市區學校代碼, 學校名稱,
    # 1年級班級數, 2年級班級數, 3年級班級數, 4年級班級數, 5年級班級數, 6年級班級數
    # 1年級男學生數, 1年級女學生數, 2年級男學生數,2年級女學生數, 3年級男學生數,3年級女學生數,
    # 4年級男學生數, 4年級女學生數, 5年級男學生數, 5年級女學生數, 6年級男學生數, 6年級女學生數,

    for year in years:
        file = os.path.join(path, f'{year}_basec.csv')
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
            except:
                df = pd.read_csv(file, encoding='BIG5')
            df = df[df['縣市名稱'].apply(lambda x: '北市' in x)]
            columns = []
            for col in df.columns:
                if '班級數' not in col and \
                        '畢業生' not in col and \
                        '教師' not in col and \
                        '職員' not in col and \
                        '代碼' not in col:
                    columns.append(col)

            df = df[columns]
            df[f'{year}_Total'] = 0
            for col in df.columns:
                if '學生數' in col and '年級' in col:
                    try:
                        df[f'{year}_Total'] += df[col].apply(lambda x: x.replace(',', '').replace('-', '0')).astype(int)
                    except:
                        df[f'{year}_Total'] += df[col]

            drops = ['1年級男學生數', '1年級女學生數', '2年級男學生數', '2年級女學生數', '3年級男學生數', '3年級女學生數']
            drops += ['4年級男學生數', '4年級女學生數', '5年級男學生數', '5年級女學生數', '6年級男學生數', '6年級女學生數']

            df.drop(columns=drops, inplace=True)

            df.to_csv(os.path.join(path, f'School_peoples_{year}.csv'), encoding='utf-8-sig', index=False)
        else:
            print(f'{file} not exist !')


def fn_gen_school_coor(path):
    # https://data.gov.tw/dataset/121225
    # https://depart.moe.edu.tw/ed4500/News_Content.aspx?n=5A930C32CC6C3818&sms=91B3AAE8C6388B96&s=B7F6EA80CA2F63EE

    files_tpe = []
    files_ntp = []
    for i, j, k in os.walk(path):
        for f in k:
            if f.endswith('.csv'):
                if '臺北市各級學校分布圖' in f:
                    files_tpe.append(f)
                if '新北市學校通訊資料' in f:
                    files_ntp.append(f)

    if len(files_tpe):
        df_tpe = pd.DataFrame()
        for f in files_tpe:
            df = pd.read_csv(os.path.join(path, f), encoding='BIG5')
            df['public'] = df['schoolname'].apply(lambda x: 0 if '私立' in x else 1)
            df['district'] = df['address'].apply(lambda x: x.split('區')[0].split('市')[-1] + '區')
            df['city'] = df['address'].apply(lambda x: '台北市')
            df_tpe = df_tpe.append(df)
        df_tpe.drop_duplicates(subset=df_tpe.columns, ignore_index=True, inplace=True)
        df_tpe.to_csv(os.path.join(path, 'School_coor.csv'), encoding='utf-8-sig', index=False)

    if len(files_ntp):
        df_tpe = pd.read_csv(os.path.join(path, 'School_coor.csv'), encoding='utf-8-sig')
        df_ntp = pd.DataFrame()
        for f in files_ntp:
            df = pd.read_csv(os.path.join(path, f), encoding='utf-8-sig')
            rename = {'alias': 'schoolname',
                      'types': 'school',
                      'postalCode': 'postalcode',
                      'phone': 'telephone',
                      'wgs84aY': 'lat',
                      'wgs84aX': 'lon',
                      }
            df.rename(columns=rename, inplace=True)
            df['public'] = df['school'].apply(lambda x: 0 if '私立' in x else 1)
            df = df[['school', 'schoolname', 'postalcode', 'address', 'telephone', 'lat', 'lon', 'public', 'district',
                     'city']]
            df_ntp = df_ntp.append(df)

        df_all = df_tpe.append(df_ntp)
        df_all = df_all[df_all['schoolname'].apply(lambda x: '進修部' not in x)]
        df_all = df_all[df_all['schoolname'].apply(lambda x: '幼兒園' not in x)]
        df_all = df_all[df_all['schoolname'].apply(lambda x: '幼稚園' not in x)]
        df_all = df_all[df_all['schoolname'].apply(lambda x: '附幼' not in x)]
        df_all = df_all[df_all['schoolname'].apply(lambda x: '補校' not in x)]

        df_all = df_all[df_all['lat'].apply(lambda x: x > 0)]
        df_all = df_all[df_all['lon'].apply(lambda x: x > 0)]

        df_all = df_all[df_all['school'] != '特教學校']
        df_all = df_all[df_all['school'] != '其他,一般']
        df_all = df_all[df_all['school'] != '一般,其他']

        df_all.drop_duplicates(subset=df_all.columns, inplace=True)
        print(f'Total {df_all.shape[0]} schools')
        df_all.to_csv(os.path.join(path, 'School_coor.csv'), encoding='utf-8-sig', index=False)


def fn_get_school_total(df, df_ps, year):
    for idx in df.index:
        s, c, d = df.loc[idx, ['schoolname', 'city', 'district']]
        k = s.split('國小')[0].split('實小')[0].split('中學附小')[0].split('附小')[0].split('實驗小學')[0].split('國民小學')[0]
        k = '私立復興國小' if k == '私立復興實驗高中' else k
        k = '臺北市立大學附小' if k == '市大' else k
        k = '私立立人國(中)小' if k == '私立立人國際國民中小學' else k
        k = '國立台北教育大學附小' if k == '國立北教大' else k

        if year >= 110:
            k = '私立復興實驗高中附設國小部' if k == '私立復興國小' else k
            k = '私立靜心高中附設國小部' if k == '私立靜心小學' else k
            k = '私立華興中學附設國小部' if k == '私立華興小學' else k

        df_d = df_ps[df_ps['縣市名稱'] == c.replace('台', '臺')].copy()
        df_d.reset_index(inplace=True)

        for i in df_d.index:
            if k in df_d.loc[i, '學校名稱']:
                total = df_d.loc[i, f'{year}_Total']
                df.at[idx, f'{year}_Total'] = total
                break

            if i == df_d.shape[0] - 1 and '小' in s and '淡水區' not in d:
                print(f'Year:{year} can NOT find peoples for {d, s, k}')

    return df


def fn_gen_school_filter(path):
    coor = os.path.join(path, 'School_coor.csv')
    df_cr = pd.read_csv(coor, encoding='utf-8-sig')
    df = df_cr[
        df_cr['address'].apply(lambda x: '台北' in str(x).replace('臺', '台') or '淡水' in str(x) or '八里' in str(x))].copy()
    df.to_csv(os.path.join(path, 'School_info.csv'), encoding='utf-8-sig', index=False)


def fn_gen_school_info(path, years):
    file = os.path.join(path, 'School_info.csv')
    if os.path.exists(file):
        df = pd.read_csv(file, encoding='utf-8-sig')
        for y in years:
            y_total = f'{y}_Total'
            y_total_file = os.path.join(path, f'School_peoples_{y}.csv')
            if y_total not in df.columns and os.path.exists(y_total_file):
                df_y = pd.read_csv(y_total_file, encoding='utf-8-sig')
                df = fn_get_school_total(df.copy(), df_y, y)

        df.sort_values(by=df.columns[-1], inplace=True, ascending=False)
        df = df[df['school'].apply(lambda x: '國小' in x)]
        df.to_csv(file, encoding='utf-8-sig', index=False)
    else:
        print(f'{file} not existed !')


def fn_gen_school_data():
    path = dic_of_path['database']
    years = range(100, 111, 1)
    fn_gen_school_coor(path)
    fn_gen_school_peoples(path, years)
    fn_gen_school_filter(path)
    fn_gen_school_info(path, years)


def fn_read_shp():
    # path = os.path.join(dic_of_path['database'], 'mapdata202112240331')
    # file = r'VILLAGE_MOI_1101214.shp'

    path = os.path.join(dic_of_path['database'], 'mapdata202104280245')
    file = r'TOWN_MOI_1100415.shp'

    g = os.path.join(path, file)
    gis_v = gpd.read_file(g, encoding='utf-8')

    gis = gis_v
    shapes = {}
    properties = {}
    for idx in gis.index:
        county = gis.loc[idx, 'COUNTYNAME']
        town = gis.loc[idx, 'TOWNNAME']
        vill = gis.loc[idx, 'VILLNAME'] if 'VILLNAME' in gis.columns else 'NA'
        # s = gis[gis.index == idx]

        if county == '臺北市':
            if 'geometry' in gis.columns:
                shapes[idx] = shape(gis.loc[idx, 'geometry'])
                properties[idx] = f'{county}, {town}, {vill}'
            else:
                assert False, f'gis cols = {gis.columns}'

    return shapes, properties


def fn_search_vill(lon, lat, shapes, properties):
    # coor_2_vill = 'Uknown1'
    vill = 'Unknown'

    for k in shapes.keys():
        # print(k, properties[k])
        if shapes[k].contains(Point(lon, lat)):
            vill = properties[k]
            # coor_2_vill = f'{lon}, {lat} is in {vill}'

            # x, y = shapes[k].exterior.xy
            # fig = plt.figure()
            # plt.plot(x, y, c="green")
            # plt.plot(lon, lat, c="red", marker='X')
            break

    return vill


def fn_coor_2_vill(lon, lat):
    shapes, properties = fn_read_shp()

    vill = fn_search_vill(lon, lat, shapes, properties)

    return vill


def fn_read_pdf(src, pages='all'):
    list_of_df = read_pdf(src, pages=pages)

    return list_of_df


def fn_gen_tax_file():
    url = 'https://www.fia.gov.tw/WEB/fia/ias/isa108s/isa108/108_165-A.pdf'
    list_of_pdf = fn_read_pdf(url)

    dic_of_dist = {
        1: '松山區',
        2: '大安區',
        3: '中正區',
        4: '萬華區',
        5: '大同區',
        6: '中山區',
        7: '文山區',
        8: '南港區',
        9: '內湖區',
        10: '士林區',
        11: '北投區',
        12: '信義區',
    }

    df_tax = pd.DataFrame()
    for p in range(len(list_of_pdf)):
        df = list_of_pdf[p]
        col_f = df.columns[0]

        if str(df[col_f][0]) == 'nan':
            for idx in df.index:
                if str(df.loc[idx, col_f]) == 'nan' or str(df.loc[idx, col_f]) == '南港區':
                    for c in range(len(df.columns) - 1):
                        df.at[idx, df.columns[c]] = df.loc[idx, df.columns[c + 1]]

        # if p == 8:
        #     print(p, len(df.columns))
        #     print(df.columns[2])
        #     print(df.iloc[:8, -2])
        df.rename(columns={'Unnamed: 0': '里'}, inplace=True)

        if len(df.columns) > 9:
            df.drop(columns=[df.columns[-1]], inplace=True)

        df_tax = pd.concat([df_tax, df], axis=0)

    df_tax.reset_index(drop=True, inplace=True)
    count = 1
    for idx in df_tax.index:
        vill = df_tax.loc[idx, '里']
        if vill.endswith('計') and count < 12:
            count += 1
        df_tax.at[idx, '行政區'] = dic_of_dist[count]

        # print(idx, vill, count, dic_of_dist[count])

    df_tax = df_tax[df_tax['里'].apply(lambda x: x.endswith('里'))].copy()
    df_tax.sort_values(by='中位數', ascending=False, inplace=True, ignore_index=True)
    df_tax.reset_index(drop=True, inplace=True)

    cols_old = df_tax.columns.tolist()
    cols = cols_old[-1:] + cols_old[:-1]
    df_tax = df_tax[cols]
    df_tax['平均_減_中位'] = df_tax['平均數'] - df_tax['中位數']

    df_tax.to_csv(os.path.join(dic_of_path['database'], '108_165-A.csv'), encoding='utf-8-sig', index=False)


def fn_web_click(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    elm.click()
    time.sleep(slp)


def fn_web_move_to(drv, act, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    act.move_to_element(elm).perform()
    time.sleep(slp)


def fn_web_send_keys(drv, val, key, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    elm.send_keys(key)
    time.sleep(slp)


def fn_web_get_text(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    txt = elm.text
    time.sleep(slp)
    return txt


def fn_web_switch(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    drv.switch_to.frame(elm)
    time.sleep(slp)


def fn_web_handle(drv, act, typ, slp, by, val, key):
    if typ == 'click':
        fn_web_click(drv, val, slp=slp, by=by)
    elif typ == 'move2':
        fn_web_move_to(drv, act, val, slp=slp, by=by)
    elif typ == 'keyin':
        fn_web_send_keys(drv, val, key, slp=slp, by=by)
    elif typ == 'getText':
        text = fn_web_get_text(drv, val, slp=slp, by=by)
        return text
    elif typ == 'iframe_switch':
        fn_web_switch(drv, val, slp=slp, by=by)
    else:
        assert False, f'Invalid web handle typ = {typ}'

    return 'NA'


def fn_web_init(link, is_headless=True):

    """
    要注意用selenium進行爬蟲的時候，
    chrome 有時候會出現「自動軟體正在控制您的瀏覽器」，然後程式可能會跑不動。
    https://ithelp.ithome.com.tw/m/articles/10267172
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option("prefs",
                                    {"profile.password_manager_enabled": False, "credentials_enable_service": False})

    if is_headless:
        options.add_argument('--headless')

    driver = webdriver.Chrome(options=options)

    driver.implicitly_wait(2)
    driver.get(link)
    action = ActionChains(driver)

    return driver, action


def fn_bc_info_parser(bc, info, dic_id, is_dbg=False):
    items = [['投資建設'],
             ['基地地址', '台北市'],
             ['預期完工', '交屋時間'],
             ['公開銷售', bc],
             ['公設比'],
             ['棟戶規劃', '樓棟規劃'],
             ['建蔽率'],
             ['樓層規劃'],
             ['車位規劃', '數量'],
             ['管理費用', '管理費'],
             ['車位配比'],
             ['結構工程', '建築構造'],
             ['基地面積'],
             ['用途規劃'],
             ['土地分區'],
             ['景觀設計'],
             ['公設設計'],
             ['燈光設計'],
             ['座向規劃'],
             ['建材說明'],
             ['建造執照'],
             ['營造公司'],
             ['建築設計'],
             ['企劃銷售']]

    infos = [_.replace(' ', '').replace('查看地圖', '').replace('有限公司', '').
                 replace('股份', '').replace('事業', '').replace('開售通知我', '') for _ in info.split('\n')]

    print(info) if is_dbg else None
    dic_of_bc_info['建案名稱'].append(bc) if bc not in dic_of_bc_info['建案名稱'] else None
    in_above = ['營造公司', '企劃銷售', '建築設計', '景觀設計', '燈光設計', '公設設計']

    for idx in range(len(infos)):
        i = infos[idx]
        for its in items:
            for it in its:
                if it in i:
                    if it == i:
                        if dic_id == 'dic1':
                            val = infos[idx + 1]
                        else:
                            val = infos[idx - 1] if i in in_above else infos[idx + 1]
                    else:
                        val = i.replace(it, '')

                    if len(dic_of_bc_info[its[0]]) < len(dic_of_bc_info['建案名稱']):
                        if '如該社區信息有誤' in val:
                            print(f'{dic_id} Bypass {it}: {val}')
                        elif its[0] == '公開銷售' and '銷' not in val and '尚未' not in val:
                            print(f'{dic_id} Bypass {it}: {val}')
                        else:
                            val = '台北市' + val if it == '台北市' else val
                            val = val.replace('造', '') if its[0] == '結構工程' and val.endswith('造') else val
                            print(bc, it, it == its[0], its[0], '-->', val) if is_dbg else None
                            dic_of_bc_info[its[0]].append(val)
                    break


def fn_get_bc_data(bc, link, dic_web_handle, is_dbg=False):
    driver, action = fn_web_init(link)
    dic_id = dic_web_handle['id'][0]
    for k in dic_web_handle.keys():
        if k == 'id':
            pass
        else:
            typ = dic_web_handle[k][0]
            slp = dic_web_handle[k][1]
            by = dic_web_handle[k][2]
            val = dic_web_handle[k][3]

            if 'web_jump' in k:
                if k == 'web_jump_2':
                    driver.get(driver.current_url + f'/{val}')
                else:
                    driver.switch_to.window(driver.window_handles[-1])
                time.sleep(slp)
                print(driver.current_url) if is_dbg else None
            else:
                try:
                    read = fn_web_handle(driver, action, typ, slp, by, val, bc)
                    if '社區概況' in read or '建築規劃' in read or '投資建設' in read or '台北市' in read:
                        fn_bc_info_parser(bc, read, dic_id, is_dbg=is_dbg)

                    if k == 'build_info' and len(dic_of_bc_info['url']) < len(dic_of_bc_info['建案名稱']):
                        print(f'{bc} --> {driver.current_url}') if is_dbg else None
                        dic_of_bc_info['url'].append(driver.current_url)
                except:
                    print(bc, dic_id, k, dic_web_handle[k][0], 'FAIL', val) if is_dbg else None
                    if k == 'bc_sel':
                        break

    driver.close()
    time.sleep(3)


def fn_get_bc_data_fr_591(bc, is_dbg=False):
    link_591 = 'https://www.591.com.tw/'

    dic_web_handle_591 = {
        'id': ['dic1'],
        'x': ['click', 1, By.XPATH, '/html/body/div[9]/div[1]/div[1]/a'],
        # 'x2': ['click', 1, By.XPATH, '/html/body/div[10]/a[1]'],
        'typ_sel': ['move2', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[1]/a[3]'],
        'city_sel': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[1]/div/span/i'],
        'city_tpe': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[1]/div/div/dl[1]/dd/a[1]'],
        'enter_bc': ['keyin', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/input'],
        'search_bt': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[2]/span'],
        'bc_sel': ['click', 1, By.XPATH, '/html/body/div[5]/div[1]/div[2]/ul/span[1]/li/a/div[2]'],
        'web_jump': ['web_jump', 2, By.XPATH, ''],
        'build_info': ['getText', 1, By.XPATH, '/html/body/div[4]/section[1]/div[2]/div[1]'],
        'build_spec_1': ['getText', 1, By.XPATH, '/html/body/div[4]/div[3]/div/div[2]'],
        'build_spec_2': ['getText', 1, By.XPATH, '/html/body/div[4]/div[2]/div/div[2]'],
        'builder_1': ['getText', 1, By.XPATH, '/html/body/div[4]/div[3]/div/div[3]'],
        'builder_2': ['getText', 1, By.XPATH, '/html/body/div[4]/div[2]/div/div[3]'],
    }

    dic_web_handle_591_2 = {
        'id': ['dic2'],
        'x': ['click', 1, By.XPATH, '/html/body/div[9]/div[1]/div[1]/a'],
        # 'x2': ['click', 1, By.XPATH, '/html/body/div[10]/a[1]'],
        'typ_sel': ['move2', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[1]/a[3]'],
        'city_sel': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[1]/div/span/i'],
        'city_tpe': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[1]/div/div/dl[1]/dd/a[1]'],
        'enter_bc': ['keyin', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/input'],
        'search_bt': ['click', 1, By.XPATH, '/html/body/section[1]/div[3]/div/div[2]/div[2]/span'],
        'addr_info': ['getText', 1, By.XPATH, '/html/body/div[5]/div[1]/div[2]/ul/span[1]/li/a/div[2]'],
        'bc_sel': ['click', 1, By.XPATH, '/html/body/div[5]/div[1]/div[2]/ul/span[1]/li/a'],
        'web_jump': ['web_jump', 3, By.XPATH, ''],
        'x4': ['click', 1, By.XPATH, '/html/body/div[1]/main/div/div[5]/div[2]/div[2]/div[3]/i'],
        'comu_info_1': ['getText', 1, By.XPATH, '/html/body/div[1]/main/div/section[3]/div'],
        'comu_info_2': ['getText', 1, By.XPATH, '/html/body/div[1]/main/div/section[4]/div'],
        'x3': ['click', 1, By.XPATH, '/html/body/div[1]/main/div/div[5]/div[2]/div[2]/div[3]/i'],
        'web_jump_2': ['web_jump', 2, By.XPATH, 'overview'],
        'build_info': ['getText', 1, By.XPATH, '/html/body/div/main/section/div[2]/div[1]'],
        'build_spec_1': ['getText', 1, By.XPATH, '/html/body/div/main/section/div[2]/div[2]'],
        'builder_1': ['getText', 1, By.XPATH, '/html/body/div/main/section/div[2]/div[3]'],
    }

    ts = time.time()
    fn_get_bc_data(bc, link_591, dic_web_handle_591, is_dbg=is_dbg)

    if bc not in dic_of_bc_info['建案名稱']:
        fn_get_bc_data(bc, link_591, dic_web_handle_591_2, is_dbg=is_dbg)

    te = time.time()
    parsing_time = int(te - ts)

    if bc in dic_of_bc_info['建案名稱']:
        dic_of_bc_info['更新日期'].append(datetime.date.today())
        dic_of_bc_info['爬蟲耗時(秒)'].append(parsing_time)
        print(f'{bc} 爬蟲耗時: {parsing_time}秒')

    for k in dic_of_bc_info.keys():
        dic_of_bc_info[k].append('NA') if len(dic_of_bc_info[k]) < len(dic_of_bc_info['建案名稱']) else None


bc_manual = [
    '長安馥',
    '菁山邸',
    '新碩大砌',
    '知鈺',
    '睿泰美',
    '御松軒',
    '信義大樓',
    '三創爵鼎',
    '樂揚煦煦',
    '長野米蘭',
    '文德好境',
    '大安滕',
    '詠大直',
    '榮耀之星',
    '中山．吉美',
    '上城若水',
    '美',
    '躍大直',
    '序東騰',
    '基泰大直',
    '世田安',
    '冠德羅斯福',
    '玖原品藏',
    '安家藏玉',
    '國雄中正',
    '紀州樂章',
    '信松廣場',
    '信義CASA',
    '天母常玉',
    '睦昇天朗',
    '岳泰峰? ',
    '力麒天沐',
    '擎天森林',
    '新美齊心岳',
    '璞知溪',
    '璞詠川',
    '城心曜曜',
    '大承',
    '築億丰盛',
    '金瑞山',
    '華固大安學府',
    '潤泰大安富陽',
    '文山晶硯',
    '玖原青',
    '天成河悅',
    '自慢藏',
    '敦北南京',
    '萬大境',
    '首傅晴海 ',
]


dic_bc_rename = {
    '中星仁愛旭': '仁愛旭',
    '宏國大道城A棟': '宏國大道城-A區',
    '宏國大道城B棟': '宏國大道城-B區',
    '宏國大道城C棟': '宏國大道城-C區',
    '宏國大道城D棟': '宏國大道城-D區',
    '政大爵鼎NO2': '政大爵鼎NO.2',
    '政大爵鼎NO1': '政大爵鼎NO.1',
    '吉美君悅': '吉美君悦',
    '忠泰衍見築': '衍見築',
    '台大學': '台太學',
    '寶舖ＣＡＲＥ': '寶舖CARE',
    '三磐舍紫II': '三磐舍紫2',
    '德杰羽森-璽': '德杰羽森',
    '德杰羽森-琚': '德杰羽森',
    '德杰羽森-玥': '德杰羽森',
    '吉吉美': '喆美',
}


def fn_get_bc_info(is_dbg=False, is_force=False, batch=10):
    # bc_manual = ['天母常玉', '奇岩綠境', '富域', '璞知溪', '國泰悠境', '皇鼎一品']
    first_bc = '康寶日出印象'
    build_case_names = bc_manual
    file_bc_names = os.path.join(dic_of_path['root'], 'pre_sold_house', 'output', 'house_all.csv')
    df_bc_names = pd.read_csv(file_bc_names, encoding='utf_8_sig', na_filter='False')
    bc_names = list(set(df_bc_names['建案名稱'].values))
    del df_bc_names
    build_case_names = build_case_names + bc_names
    build_case_names.remove(first_bc) if first_bc in build_case_names else None
    build_case_names = [first_bc] + list(set(build_case_names))

    assert len(build_case_names) > batch, f'bc num {len(build_case_names)} < batch size {batch} !'

    file = os.path.join(dic_of_path['database'], 'build_case_info.csv')
    bc_parsed = []
    if os.path.exists(file):
        df_pre = pd.read_csv(file, encoding='utf_8_sig', na_filter='False')
        bc_parsed = df_pre['建案名稱'].values

    print(f'There are {len(build_case_names)} build cases to parsing')

    # bc = dic_bc_rename[bc] if bc in dic_bc_rename.keys() else bc

    build_case_names = [dic_bc_rename[bc] if bc in dic_bc_rename.keys() else bc for bc in build_case_names]

    for bc in build_case_names:

        print(f'\n({build_case_names.index(bc) + 1}/{len(build_case_names)}), {bc} \n=================\n')
        if bc in bc_parsed and is_force is False:
            print(f'{bc} already existed in {file}')
        else:
            if '建案名稱' in dic_of_bc_info.keys():
                if bc in dic_of_bc_info['建案名稱']:
                    print(f'{bc} already in dic_of_bc_info')
                else:
                    fn_get_bc_data_fr_591(bc, is_dbg=is_dbg)
            else:
                fn_get_bc_data_fr_591(bc, is_dbg=is_dbg)

            print(f'{bc}: {len(dic_of_bc_info.keys())} cols') if is_dbg else None

        if len(dic_of_bc_info['建案名稱']) > 0 and len(dic_of_bc_info['建案名稱']) % batch == 0:
            try:
                df_bc_info = pd.DataFrame(dic_of_bc_info)
            except:
                for k in dic_of_bc_info:
                    print(k, len(dic_of_bc_info[k]))
                df_bc_info = pd.DataFrame(dic_of_bc_info)

            if os.path.exists(file):
                df_old = pd.read_csv(file, encoding='utf_8_sig', na_filter=False)
                df_new = pd.concat([df_old, df_bc_info], axis=0, ignore_index=True)
                df_new.drop_duplicates(subset=df_new.columns, inplace=True)
                df_new.to_csv(file, encoding='utf_8_sig', index=False)
                print(f'Save {len(dic_of_bc_info["建案名稱"])} bc data to csv {df_new.shape}')
            else:
                df_bc_info.to_csv(file, encoding='utf_8_sig', index=False)

            for k in dic_of_bc_info.keys():
                dic_of_bc_info[k] = []

    if '建案名稱' in dic_of_bc_info.keys():
        if len(dic_of_bc_info['建案名稱']) > 0:
            df_bc_info = pd.DataFrame(dic_of_bc_info)
            df_old = pd.read_csv(file, encoding='utf_8_sig', na_filter=False)
            df_new = pd.concat([df_old, df_bc_info], axis=0, ignore_index=True)
            df_new.drop_duplicates(subset=df_new.columns, inplace=True)
            df_new.to_csv(file, encoding='utf_8_sig', index=False)
            print(f'Save {len(dic_of_bc_info["建案名稱"])} bc data to csv {df_new.shape} done !')


def fn_gen_households(x):
    if x == 'NA':
        count = 0
    else:
        total = x.split('，')
        households = [_ for _ in total if '戶' in _]
        numbers = [int(_.split('戶')[0]) for _ in households]
        count = sum(numbers)

    return count


def fn_get_flat_parking_num(x):
    num = 0
    kw = '平面式'
    if '、' in x:
        pks = x.split('、')
        for pk in pks:
            if kw in pk:
                num = int(pk.replace(kw, '').replace('個', ''))
                break
    else:
        if kw in x:
            num = int(x.replace(kw, '').replace('個', ''))

    return num


def fn_get_mach_parking_num(x):
    num = 0
    kw = '機械式'
    if '、' in x:
        pks = x.split('、')
        for pk in pks:
            if kw in pk:
                num = int(pk.replace(kw, '').replace('個', ''))
                break
    else:
        if kw in x:
            num = int(x.replace(kw, '').replace('個', ''))

    return num


def fn_get_build_cap(v):

    # 第三種住宅區 第三之二種住宅區,
    # 第三種商業區
    # 特定農業區甲種建築用地
    # 第三種商業區，第四種住宅區
    k = 'NA'
    x = v.split('（')[0].split('(')[0].split('、')[0].split('，')[0]
    if x.endswith('區') and '種' in x:
        k = x.split('種')[-1][0]+x.split('種')[0].replace('第', '')
    else:

        '''
        '住ㄧ': 60,
        '住二': 120,
        '住二之一': 160,
        '住二之二': 225,
        '住三': 225,
        '住三之一': 300,
        '住三之二': 400,
        '住四': 300,
        '住四之一': 400,
        '商ㄧ': 360,
        '商二': 630,
        '商三': 560,
        '商四': 800,
        '工二': 200,
        '工三': 300,
        '''

        for typ in dic_of_capacity_ratio.keys():

            if '之' in x:
                if '之' in typ:
                    if typ in x:
                        k = typ
                        break
            else:
                if typ in x:
                    k = typ
                    break

        if k == 'NA':
            if '住3' in x:
                k = '住三'
            else:
                print(x, 'Unknown Cap. format')

    if k in dic_of_capacity_ratio.keys():
        print(v, '-->', k)
        cap = dic_of_capacity_ratio[k]
    else:
        # print(k, 'Not in dic_of_capacity_ratio.keys()')
        cap = x

    return cap


def fn_gen_bc_info_extend():
    csv_file = 'build_case_info.csv'
    df = pd.read_csv(os.path.join(dic_of_path['database'], csv_file), encoding='utf_8_sig', na_filter=False)
    df.drop_duplicates(subset=df.columns, inplace=True)

    df['行政區'] = df['基地地址'].apply(lambda x: x if x == 'NA' else str(x).split('區')[0].split('北市')[-1] + '區')
    df['建照年度'] = df['建造執照'].apply(lambda x: x if x == 'NA' else str(x).split('建字')[0])

    #  地上15、24層，地下5層
    df['地上樓層'] = df['樓層規劃'].apply(
        lambda x: x if x == 'NA' else str(x).split('層，地下')[0].split('層,地下')[0].replace('地上', ''))
    df['地上樓層'] = df['地上樓層'].apply(lambda x: x.replace('層', '') if '層' in x else x)

    df['地下樓層'] = df['樓層規劃'].apply(
        lambda x: x if x == 'NA' else str(x).split('層，地下')[-1].split('層,地下')[-1].replace('層', ''))
    df['地下樓層'] = df['地下樓層'].apply(lambda x: x.replace('層', '') if '層' in x else x)

    df['總戶數'] = df['棟戶規劃'].apply(fn_gen_households)
    df['車位配比'] = df['車位配比'].astype(str)
    df['平面車位'] = df['車位規劃'].apply(fn_get_flat_parking_num)
    df['機械車位'] = df['車位規劃'].apply(fn_get_mach_parking_num)
    df['總車位數'] = df['平面車位'] + df['機械車位']
    df['公開銷售'] = df['公開銷售'].apply(lambda x: x.split('已完銷')[-1] + '已完銷' if '已完銷' in x else x)
    df['棟數'] = df['棟戶規劃'].apply(
        lambda x: int(x.split('棟')[0].split('，')[-1]) if '，' in x.split('棟')[0] else int(x.split('棟')[0]))

    s = df['建蔽率'].apply(lambda x: float(x.replace('%', 'e-2')))
    f = df['地上樓層'].apply(lambda x: int(max(x.split('、'))) if '、' in x else int(x))

    # df['容積率(%)'] = s * f
    # df['容積率(%)'] = df['容積率(%)'].apply(lambda x: int(100 * x))

    df['容積率(%)'] = df['土地分區'].apply(fn_get_build_cap)

    df['基地面積(坪)'] = df['基地面積'].apply(lambda x: round(float(x.replace('坪', '')), 0))
    df['建蔽率(%)'] = df['建蔽率'].apply(lambda x: float(x.replace('%', '')))

    # df['公設比(%)'] = df['公設比'].apply(
    #     lambda x: float(x.split('~')[-1].replace('%', '') if '~' in x else float(x.replace('%', ''))))

    df['公設比(%)'] = df['公設比'].apply(
        lambda x: x if '暫無' in x else x.split('~')[-1].replace('%', ''))

    df['建蔽面積(坪)'] = df['基地面積(坪)'] * df['建蔽率(%)'] / 100
    df['建蔽面積(坪)'] = df['建蔽面積(坪)'].astype(int)
    df['完工年度'] = df['預期完工'].apply(lambda x: int(x.split('年')[0]) - 1911 if '年' in x else x)
    df['預估工期'] = df['完工年度'].astype(str)
    for i in range(df['完工年度'].shape[0]):
        fr = df['建照年度'].values[i]
        to = df['完工年度'].values[i]
        if 'NA' in fr or '隨時' in str(to):
            df['預估工期'].values[i] = 'NA'
        else:
            df['預估工期'].values[i] = int(to) - int(fr)

    for idx in df['建照年度'].index:
        v1 = df.loc[idx, '建照年度']
        v2 = df.loc[idx, '完工年度']
        try:
            df.at[idx, '建照年度_tmp'] = int(v1)
        except:
            v2 = datetime.datetime.now().year if '隨時交屋' == v2 or 'NA' == v2 else v2
            df.at[idx, '建照年度_tmp'] = int(v2) - 5

        df.at[idx, '完工年度_tmp'] = datetime.datetime.now().year if '隨時交屋' == v2 else int(v2)

    df.sort_values(by=['行政區', '建照年度_tmp', '完工年度_tmp', '基地面積(坪)'], ascending=[True, False, True, False], inplace=True,
                   ignore_index=True)

    cols_order = ['行政區', '建照年度', '完工年度', '公開銷售', '建案名稱', '基地面積(坪)', '建蔽面積(坪)', '建蔽率(%)', '容積率(%)',
                  '公設比(%)', '棟數', '地上樓層', '地下樓層', '總戶數', '平面車位', '機械車位', '總車位數', '預估工期', '建造執照',
                  '預估工期', '投資建設', '建築設計', '營造公司', '企劃銷售', '結構工程', '座向規劃', '車位規劃', '車位配比',
                  '用途規劃', '土地分區', '管理費用', '景觀設計', '公設設計', '燈光設計', '棟戶規劃', '樓層規劃', '基地地址',
                  '建材說明', 'url', '爬蟲耗時(秒)', '更新日期']

    df = df[cols_order]
    df = df[df['行政區'] != 'NA']
    df.to_csv(os.path.join(dic_of_path['database'], csv_file.split('.csv')[0] + '_ext.csv'), encoding='utf_8_sig',
              index=False)


def fn_gen_litigation():
    link = 'https://law.judicial.gov.tw/FJUD/default.aspx'

    dic_web_handle_litigation = {
        'enter_builder': ['keyin', 1, By.XPATH, '/html/body/form/div[5]/div/div[1]/table/tbody/tr/td/div[1]/input'],
        'search_bt': ['click', 1, By.XPATH, '/html/body/form/div[5]/div/div[1]/table/tbody/tr/td/div[1]/div/input[3]'],
        'by_yr_bt': ['click', 1, By.XPATH, '/html/body/form/div[5]/div/div[2]/div/div[1]/div[2]/div[2]/div[1]/div/a'],
        'yr_evt': ['getText', 2, By.XPATH, '/html/body/form/div[5]/div/div[2]/div/div[1]/div[2]/div[2]/div[2]/div/ul'],
        'iframe_switch': ['iframe_switch', 1, By.XPATH, '/html/body/form/div[5]/div/div[3]/iframe'],
        'clz_bt': ['click', 1, By.XPATH, '/html/body/form/div[3]/div/div[1]/a'],
        'detail': ['getText', 2, By.XPATH, '/html/body/form/div[3]/div/table/tbody'],
    }

    # builders = ['康寶建設', '勝岳營造', '華固建設']

    df_bc_name = pd.read_csv(os.path.join(dic_of_path['database'], 'build_case_info_ext.csv'), encoding='utf_8_sig', na_filter=False)
    builders = list(set(df_bc_name['投資建設'].values))
    constructor = list(set(df_bc_name['營造公司'].values))
    seller = list(set(df_bc_name['企劃銷售'].values))

    search = []
    # sep = ['、', '、', '-', '/', 'X', '(']
    for b in list(builders+constructor):
        if '、' in b:
            for each_b in list(b.split('、')):
                search.append(each_b)
        elif ',' in b:
            for each_b in list(b.split(',')):
                search.append(each_b)
        elif '-' in b:
            for each_b in list(b.split('-')):
                search.append(each_b)
        elif '/' in b:
            for each_b in list(b.split('/')):
                search.append(each_b)
        elif 'X' in b:
            for each_b in list(b.split('X')):
                search.append(each_b)
        elif '(' in b:
            for each_b in list(b.split('(')):
                search.append(each_b.replace(')', ''))
        else:
            search.append(b)

    search = list(set(search))
    search = [s.replace(')', '').replace('建築團隊', '建築').replace('建設開發', '建設').replace('TSEC', '') for s in search]
    search.remove('待定')
    search.remove('暫無')

    # print(len(search), search)

    df_lg = pd.DataFrame()

    for b in search:
        evts = 0
        dic_detail = defaultdict(list)
        driver, action = fn_web_init(link, is_headless=True)

        for k in dic_web_handle_litigation.keys():
            # print(b, k)
            typ = dic_web_handle_litigation[k][0]
            slp = dic_web_handle_litigation[k][1]
            by = dic_web_handle_litigation[k][2]
            val = dic_web_handle_litigation[k][3]

            try:
                read = fn_web_handle(driver, action, typ, slp, by, val, b)
            except:
                print(f'{b} --> 查無裁判案件')
                break

            if read == 'NA':
                pass
            else:
                reads = read.split('\n')
                if k == 'yr_evt':
                    for r in reads:
                        try:
                            evts += int(r)
                        except:
                            pass
                elif k == 'detail':
                    for i, v in enumerate(reads):
                        if i > 0:
                            dic_detail['建商營造'].append(b)
                            dic_detail['裁判字號'].append(v.split('（')[0].split('. ')[-1])
                            dic_detail['裁判日期'].append(v.split('） ')[-1].split(' ')[0])
                            dic_detail['裁判案由'].append(v.split(' ')[-1])

                    df = pd.DataFrame(dic_detail)
                    df['歷年案件'] = evts
                    df_lg = pd.concat([df_lg, df], axis=0)
                    dic_detail.clear()

                else:
                    print(k, '\n', read)

        print(f'({search.index(b)+1}/{len(search)})', b, '-->', evts, 'cases') if k == 'detail' else None

        driver.close()
        driver.quit()
        time.sleep(1)

    df_lg = df_lg[['建商營造', '歷年案件', '裁判日期', '裁判案由', '裁判字號']]
    df_lg.reset_index(drop=True, inplace=True)
    df_lg.to_csv(os.path.join(dic_of_path['database'], 'builder_litigation.csv'), encoding='utf-8-sig', index=False)


def fn_main():
    # fn_gen_mrt_coor()
    # fn_get_travel_time()
    # fn_get_mrt_tput('進站', 6, 10)
    # fn_get_mrt_tput('出站', 6, 10)
    # fn_gen_school_data()
    # fn_gen_tax_file()
    pass

    fn_get_bc_info(is_dbg=False, is_force=False, batch=10)
    fn_gen_bc_info_extend()

    fn_gen_litigation()


if __name__ == '__main__':
    fn_main()
