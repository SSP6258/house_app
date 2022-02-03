import time
import datetime
import os
import numpy as np
import pandas as pd
import cn2an
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from geopy.distance import geodesic
from collections import defaultdict
from workalendar.asia import Taiwan

dic_of_path = {
    # 'root': r'D:\05_Database\house_data',
    # 'database': r'D:\05_Database\house_data\database'
    'root': r'house_data',
    'database': r'house_data/database',
}


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


def fn_get_admin_dist(addr):
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


def fn_get_coor_fr_db(addr, df_coor):
    dic_of_dist = fn_get_admin_dist(addr)

    for k, v in dic_of_dist.items():
        if v != 'NA' and k in df_coor.columns and v in df_coor[k].values:
            df_coor = df_coor[df_coor[k] == v]
        # print(k, v, v in df_coor[k].values, df_coor.shape[0])

    sel = 0
    matched = False
    if df_coor.shape[0]:
        for a in ['號', '弄', '巷', '路']:
            if a in dic_of_dist.keys() and a in dic_of_dist[a] and not matched:
                num = int(dic_of_dist[a].split(a)[0])
                nums = df_coor[a].apply(
                    lambda x: x if str(x) == 'nan' else int(str(x).split(a)[0].split('之')[0])).tolist()
                diff = [abs(n - num) for n in nums if str(n) != 'nan']
                sel = diff.index(min(diff))
                matched = True
                print(a, num, nums, sel, nums[sel], matched)
                break

    coor = df_coor[['lat', 'lon']].iloc[sel, :]
    coor = tuple(coor)
    addr_match = df_coor.index[sel]

    if matched:
        print('Coor From DB: ', addr, ' --> ', addr_match, coor)
    else:
        print(f'can NOT find similar addr of {addr}')

    return coor


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

    if addr in df_addr_coor.index:
        lat = round(df_addr_coor.loc[addr]['lat'], 5)
        lon = round(df_addr_coor.loc[addr]['lon'], 5)
        addr_coor = (lat, lon)
        print(addr, '--> known coor: ', addr_coor)
    else:
        try:
            addr_coor = fn_get_coordinate(addr, slp)
            # print(addr, addr_coor)
        except:
            print(f'find {addr} from database')
            addr_coor = fn_get_coordinate(addr, slp)
            addr_coor = fn_get_coor_fr_db(addr, df_addr_coor.copy())
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
    mrt = mrt_s.replace('站', '')
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

    return geo_info, is_save


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

        df_d = df_ps[df_ps['縣市名稱'] == c.replace('台', '臺')].copy()
        df_d.reset_index(inplace=True)

        for i in df_d.index:
            if k in df_d.loc[i, '學校名稱']:
                total = df_d.loc[i, f'{year}_Total']
                df.at[idx, f'{year}_Total'] = total
                break

            if i == df_d.shape[0] - 1 and '小' in s:
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
    years = range(100, 110, 1)
    fn_gen_school_coor(path)
    fn_gen_school_peoples(path, years)
    fn_gen_school_filter(path)
    fn_gen_school_info(path, years)


def fn_main():
    # fn_gen_mrt_coor()
    # fn_get_travel_time()
    # fn_get_mrt_tput('進站', 6, 10)
    # fn_get_mrt_tput('出站', 6, 10)
    fn_gen_school_data()


if __name__ == '__main__':
    fn_main()
