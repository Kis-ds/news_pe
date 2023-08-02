### hyperlink_test.py확인
#### 업데이트 확인용


import json
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
import datetime


def loadPickle(path):
    with open(path, 'rb') as fr:
        df = pickle.load(fr)
    return df


# JSON을 읽어 들이는 함수. (lottie 안되면 변경)
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res


def load_lottie():
    with open('./resources/news_lottie_4.json', 'r', encoding='utf-8-sig') as st_json:
        return json.load(st_json)


# # test about link
def make_clickable(link):
    # text = link.split('=')[1]
    return f'<a target="aboutlink" href="{link}">{"바로가기"}</a>'



def return_key_list(texts):
    return texts.replace(' ', '').replace('[', '').replace(']', '').replace('"', '').split(',')


################ 환경 설정 ################


# DATA LOAD
df = loadPickle('./resources/all_full_db.pkl')  # all_full_db_060708




ipo_search_list = ["상장 연기","상장 철회","프리 IPO","기술특례상장","상장예비심사","IPO 주관","IPO 수요예측"]
pe_search_list = ["사모펀드","경영권 인수","공개매수","M&A","블록딜","지배구조개선","소수지분", "일감몰아주기", "LP Limited Partner", "컨소시엄", "볼트온", "롤업", "롤링베이시스", "LOI", "투자유치"]
company_search_list = ["회사채","회사채 수요예측","회사채 발행","회사채 공모","유상증자","메자닌","영구채"]

# 제목 일자 설정
df['수집일자'] = df['크롤링일자'].dt.strftime('%Y-%m-%d')
crawling_date = df['수집일자'].max()
yester_date = df.크롤링일자.max() - pd.Timedelta(1, unit='days')


# 일단 여기 주석
# 요약키워드 리스트만들기
# df['요약키워드리스트'] = df.요약키워드.apply(return_key_list)
# all_keyword = []
# for i in df.요약키워드리스트:
#     all_keyword += i
# key_dict = collections.Counter(all_keyword)

# Style
st.set_page_config(layout="wide")

# Style READ
with open('style2.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

### # ###### web layout #######


# LOGO(lottie) AND TITLE
empty_head, col_lottie, col_t2itle = st.columns([0.3, 0.5, 1])
with empty_head:
    # 여백
    st.empty()

with col_lottie:
    # 기존 LOTTIE READ
    lottie = load_lottie()
    st_lottie(lottie, speed=1, loop=True, width=250, height=250, )

with col_t2itle:
    ''
    st.title('뉴스 데이터 한눈에 보기')
    st.write(':red[일별 업데이트 (매일 7시 반 기준)]')

# 세션 상태 초기화 ( id / password)
if 'client_id' not in st.session_state:
    st.session_state['client_id'] = ''
if 'client_secret' not in st.session_state:
    st.session_state['client_secret'] = ''

# 세션 초기화 상태시 DATA 추가
if 'data2' not in st.session_state:
    st.session_state['data2'] = ''

st.session_state['data2'] = df

###### side bar ######

# client_settings side bar 작성 (사이드바 에서 id, scret입력받기)
with st.sidebar.form(key='client_settins', clear_on_submit=True):
    st.header('네이버 API 로그인하기')

    # input
    client_id = st.text_input('client_id :', value=st.session_state['client_id'])
    client_secret = st.text_input('secret:', type='password', value=st.session_state['client_secret'])
    if st.form_submit_button(label='login'):
        st.session_state['client_id'] = client_id
        st.session_state['client_secret'] = client_secret
        st.experimental_rerun()

###### main form #####

st.write("---")
st.write(f'{crawling_date}일 기준 수집된 전체 주요 기사는 {df.shape[0]}개 입니다.')

st.subheader('* 검색어 TOP 5', help="당일 수집된 검색어별 기사 수를 집계한 결과이며,\n 전일자의 기사수와 어느 정도 차이나는지 확인할 수 있습니다. ")

# dataframe 만들기
keyword_today = df[df.수집일자 == crawling_date].검색어.value_counts().reset_index()
keyword_today.columns = ['key_', 't_cnt']
keyword_yesterday = df[df.크롤링일자 == yester_date].검색어.value_counts().reset_index()
keyword_yesterday.columns = ['key_', 'y_cnt']
top_today = pd.merge(keyword_today, keyword_yesterday, on='key_', how='left').fillna(0)
top_today['delta_cnt'] = top_today.t_cnt - top_today.y_cnt
top_today['delta_cnt'] = top_today.delta_cnt.astype(float)
# top_today = top_today.astype({'delta_cnt':'int'})


# top_yesterday해서 가져오기 → delta 값 수정
empty_p1_1, col_p1_1, col_p1_2, col_p1_3, col_p1_4, col_p1_5, empty_p1_2 = st.columns(
    [0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2])
with empty_p1_1:
    st.empty()

with col_p1_1:
    st.metric(label=top_today['key_'][0], value=top_today['t_cnt'][0], delta=top_today['delta_cnt'][0],
              delta_color="inverse")
with col_p1_2:
    st.metric(label=top_today['key_'][1], value=top_today['t_cnt'][1], delta=top_today['delta_cnt'][1],
              delta_color="inverse")
with col_p1_3:
    st.metric(label=top_today['key_'][2], value=top_today['t_cnt'][2], delta=top_today['delta_cnt'][2],
              delta_color="inverse")
with col_p1_4:
    st.metric(label=top_today['key_'][3], value=top_today['t_cnt'][3], delta=top_today['delta_cnt'][3],
              delta_color="inverse")
with col_p1_5:
    st.metric(label=top_today['key_'][4], value=top_today['t_cnt'][4], delta=top_today['delta_cnt'][4],
              delta_color="inverse")

with empty_p1_2:
    st.empty()

''
'---'
''

##### IPO PART
len_2_1 = df[df['검색어'].isin(ipo_search_list)].shape[0]


true_type_1 = df[(df.중복기업기사제거 == False) & (df['검색어'].isin(ipo_search_list))][
    ['검색어', '제목내최초기업명', '타이틀', '발행일시',  '뉴스URL']].reset_index(drop=True)

# 중복제거x
IPO_df = df[(df.토픽 == 'IPO')][['검색어', '제목내최초기업명', '타이틀','발행일시',  '뉴스URL']]


col_p2_1, empty_p2_1, empty_p2_2, empty_p2_3, empty_p2_4, col_p2_2 = st.columns([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
with col_p2_1:
    st.subheader('* IPO 동향')
    st.empty()
with empty_p2_1:
    st.empty()
with empty_p2_2:
    st.empty()
with empty_p2_3:
    st.empty()
with empty_p2_4:
    st.empty()
with col_p2_2:
    skipped_button_IPO = st.checkbox('기업 중복기사 생략', help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. " +
                                                        "\n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.")

##### css 적용한 기업top
''
empty_t3, col_t2, empty_t2 = st.columns([0.1, 0.6, 0.1])
with empty_t3:
    st.empty()
with col_t2:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        st.write(':red[▶  전일 대비 조회]')

        today = df[(df.수집일자 == crawling_date) & (df.토픽 == 'IPO')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today.columns = ['제목내최초기업명', 'today_cnt']
        yester = df[(df.크롤링일자 == yester_date) & (df.토픽 == 'IPO')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day = pd.merge(today, yester, on='제목내최초기업명', how='left').fillna(0)
        company_2day['delta_cnt'] = company_2day.today_cnt - company_2day.yester_cnt

        today_shape = company_2day.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                '제목내최초기업명': ['ㅡ'] * make_null_cnt,
                'today_cnt': [0] * make_null_cnt,
                'yester_cnt': [0] * make_null_cnt,
                'delta_cnt': [0] * make_null_cnt})

            company_2day = pd.concat([company_2day, null_df]).reset_index(drop=True)



        col_lottie1, col_t2itle2, col33, col44, col55 = st.columns(5)
        # st.dataframe(company_2day)
        with col_lottie1:
            st.metric(label=company_2day['제목내최초기업명'][0], value=company_2day['today_cnt'][0],
                      delta=company_2day['delta_cnt'][0])
        with col_t2itle2:
            st.metric(label=company_2day['제목내최초기업명'][1], value=company_2day['today_cnt'][1],
                      delta=company_2day['delta_cnt'][1])
        with col33:
            st.metric(label=company_2day['제목내최초기업명'][2], value=company_2day['today_cnt'][2],
                      delta=company_2day['delta_cnt'][2])
        with col44:
            st.metric(label=company_2day['제목내최초기업명'][3], value=company_2day['today_cnt'][3],
                      delta=company_2day['delta_cnt'][3])
        with col55:
            st.metric(label=company_2day['제목내최초기업명'][4], value=company_2day['today_cnt'][4],
                      delta=company_2day['delta_cnt'][4])

    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')

        ### 7day 수정 확인
        sort_ipo_1 = df[df.토픽 == 'IPO'][['제목내최초기업명', '크롤링일자']].dropna(axis=0).groupby(['제목내최초기업명']).max().reset_index()
        sort_ipo_2 = pd.merge(sort_ipo_1, df[df.토픽 == 'IPO'][['제목내최초기업명', '크롤링일자', '기업기준동일기사개수_일주일']].dropna(axis=0).drop_duplicates(), on=['제목내최초기업명', '크롤링일자'], how='left')
        company_7day = sort_ipo_2.sort_values(by='기업기준동일기사개수_일주일', ascending=False).reset_index(drop=True).drop(['크롤링일자'], axis=1)


        company_7day.columns = ['key_', 'cnt']

        # 예외 처리
        today_shape = company_7day.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                'key_': ['ㅡ'] * make_null_cnt,
                'cnt': [0] * make_null_cnt})

            company_7day = pd.concat([company_7day, null_df]).reset_index(drop=True)

        col_lottie1, col_t2itle2, col33, col44, col55 = st.columns(5)
        with col_lottie1:
            st.metric(label=company_7day['key_'][0], value=company_7day['cnt'][0], delta='')
        with col_t2itle2:
            st.metric(label=company_7day['key_'][1], value=company_7day['cnt'][1], delta='')
        with col33:
            st.metric(label=company_7day['key_'][2], value=company_7day['cnt'][2], delta='')
        with col44:
            st.metric(label=company_7day['key_'][3], value=company_7day['cnt'][3], delta='')
        with col55:
            st.metric(label=company_7day['key_'][4], value=company_7day['cnt'][4], delta='')
with empty_t2:
    st.empty()
''


radio_sel1_3 = st.multiselect(f"수집 뉴스 조회", ipo_search_list, default=ipo_search_list, key='part2_1', max_selections=7)


if skipped_button_IPO:
    true_type_1['뉴스URL'] = true_type_1['뉴스URL'].apply(make_clickable)
    tmp1 = true_type_1[(true_type_1['검색어'].isin(radio_sel1_3))].reset_index(drop=True)
    tmp1.columns = ['검색어', '기업명', '타이틀', '발행일시', '뉴스URL']
    tmp1 = tmp1.to_html(escape=False)
    st.write(tmp1, unsafe_allow_html=True)

else:
    IPO_df['뉴스URL'] = IPO_df['뉴스URL'].apply(make_clickable)
    tmp2 = IPO_df[(IPO_df['검색어'].isin(radio_sel1_3))].reset_index(drop=True)
    tmp2.columns = ['검색어', '기업명', '타이틀', '발행일시', '뉴스URL']
    tmp2 = tmp2.to_html(escape=False)
    st.write(tmp2, unsafe_allow_html=True)


'-----'




##### PE PART
len_2_2 = df[df['검색어'].isin(pe_search_list)].shape[0]
true_type_2 = df[(df.중복기업기사제거 == False) & (df['토픽'] == 'PE')][['검색어', '제목내최초기업명', '타이틀', '발행일시', '뉴스URL']].reset_index(drop=True)
PE_df = df[(df.토픽 == 'PE')][['검색어', '제목내최초기업명', '타이틀','발행일시',  '뉴스URL']]

col_p3_1, empty_p3_1, empty_p3_2, empty_p3_3, empty_p3_4, col_p3_2 = st.columns([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
with col_p3_1:
    st.subheader('* PE 동향')
with empty_p3_1:
    st.empty()
with empty_p3_2:
    st.empty()
with empty_p3_3:
    st.empty()
with empty_p3_4:
    st.empty()
with col_p3_2:
    skipped_button_PE = st.checkbox('기업 중복기사 생략',
                                    help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. \n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.",
                                    key='PE_BUTTON')
    ''

empty_t3, col_t2, empty_t4 = st.columns([0.1, 0.6, 0.1])
with empty_t3:
    st.empty()
with col_t2:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        st.write(':red[▶  전일 대비 조회]')

        today_PE = df[(df.수집일자 == crawling_date) & (df.토픽 == 'PE')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today_PE.columns = ['제목내최초기업명', 'today_cnt']
        yester_PE = df[(df.크롤링일자 == yester_date) & (df.토픽 == 'PE')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester_PE.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day_PE = pd.merge(today_PE, yester_PE, on='제목내최초기업명', how='left').fillna(0)
        company_2day_PE['delta_cnt'] = company_2day_PE.today_cnt - company_2day_PE.yester_cnt

        # 예외 처리
        today_shape = company_2day_PE.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                '제목내최초기업명': ['ㅡ'] * make_null_cnt,
                'today_cnt': [0] * make_null_cnt,
                'yester_cnt': [0] * make_null_cnt,
                'delta_cnt': [0] * make_null_cnt})

            company_2day_PE = pd.concat([company_2day_PE, null_df]).reset_index(drop=True)


        col_title_top1_pe, col_title_top2_pe, col_title_top3_pe, col_title_top4_pe, col_title_top5_pe = st.columns(5)
        with col_title_top1_pe:
            st.metric(label=company_2day_PE['제목내최초기업명'][0], value=company_2day_PE['today_cnt'][0],
                      delta=company_2day_PE['delta_cnt'][0])
        with col_title_top2_pe:
            st.metric(label=company_2day_PE['제목내최초기업명'][1], value=company_2day_PE['today_cnt'][1],
                      delta=company_2day_PE['delta_cnt'][1])
        with col_title_top3_pe:
            st.metric(label=company_2day_PE['제목내최초기업명'][2], value=company_2day_PE['today_cnt'][2],
                      delta=company_2day_PE['delta_cnt'][2])
        with col_title_top4_pe:
            st.metric(label=company_2day_PE['제목내최초기업명'][3], value=company_2day_PE['today_cnt'][3],
                      delta=company_2day_PE['delta_cnt'][3])
        with col_title_top5_pe:
            st.metric(label=company_2day_PE['제목내최초기업명'][4], value=company_2day_PE['today_cnt'][4],
                      delta=company_2day_PE['delta_cnt'][4])

    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')


        ### 7day 수정 확인
        sort_pe_1 = df[df.토픽 == 'PE'][['제목내최초기업명', '크롤링일자']].dropna(axis=0).groupby(['제목내최초기업명']).max().reset_index()
        sort_pe_2 = pd.merge(sort_pe_1, df[df.토픽 == 'PE'][['제목내최초기업명', '크롤링일자', '기업기준동일기사개수_일주일']].dropna(axis=0).drop_duplicates(), on=['제목내최초기업명', '크롤링일자'], how='left')
        company_7day_PE = sort_pe_2.sort_values(by='기업기준동일기사개수_일주일', ascending=False).reset_index(drop=True).drop(['크롤링일자'], axis=1)




        company_7day_PE.columns = ['key_', 'cnt']

        # 예외 처리
        today_shape = company_7day_PE.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                'key_': ['ㅡ'] * make_null_cnt,
                'cnt': [0] * make_null_cnt})

            company_7day_PE = pd.concat([company_7day_PE, null_df]).reset_index(drop=True)

        col_title_top1_pe_7, col_title_top2_pe_7, col_title_top3_pe_7, col_title_top4_pe_7, col_title_top5_pe_7 = st.columns(
            5)
        with col_title_top1_pe_7:
            st.metric(label=company_7day_PE['key_'][0], value=company_7day_PE['cnt'][0], delta='')
        with col_title_top2_pe_7:
            st.metric(label=company_7day_PE['key_'][1], value=company_7day_PE['cnt'][1], delta='')
        with col_title_top3_pe_7:
            st.metric(label=company_7day_PE['key_'][2], value=company_7day_PE['cnt'][2], delta='')
        with col_title_top4_pe_7:
            st.metric(label=company_7day_PE['key_'][3], value=company_7day_PE['cnt'][3], delta='')
        with col_title_top5_pe_7:
            st.metric(label=company_7day_PE['key_'][4], value=company_7day_PE['cnt'][4], delta='')
with empty_t4:
    st.empty()
''
radio_sel2_3 = st.multiselect(f"수집 뉴스 조회", pe_search_list, default=pe_search_list, key='part2_2', max_selections=len(pe_search_list))



if skipped_button_PE:
    true_type_2['뉴스URL'] = true_type_2['뉴스URL'].apply(make_clickable)
    tmp2_2 = true_type_2[(true_type_2['검색어'].isin(radio_sel2_3))].reset_index(drop=True)#.to_html(escape= False)
    tmp2_2.columns = ['검색어', '기업명', '타이틀', '발행일시', '뉴스URL']
    tmp2_2 = tmp2_2.to_html(escape=False)
    st.write(tmp2_2, unsafe_allow_html=True)


else:
    PE_df['뉴스URL'] = PE_df['뉴스URL'].apply(make_clickable)
    tmp2_3 = PE_df[(PE_df['검색어'].isin(radio_sel2_3))].reset_index(drop=True)#.to_html(escape=False)
    tmp2_3.columns = ['검색어', '기업명', '타이틀', '발행일시', '뉴스URL']
    tmp2_3 = tmp2_3.to_html(escape=False)
    st.write(tmp2_3, unsafe_allow_html=True)



'-----'






##### 회사채 PART

len_2_3 = df[df['검색어'].isin(company_search_list)].shape[0]

# 중복제거 VER
true_type_3 = df[(df.중복기업기사제거 == False) & (df['검색어'].isin(company_search_list))][['검색어','제목내최초기업명', '타이틀','발행일시', '뉴스URL']].reset_index(drop=True)

# 중복제거X
COM_df = df[(df.토픽 == '회사채')][['검색어', '제목내최초기업명', '타이틀', '발행일시', '뉴스URL' ]]

col_p4_1, empty_p4_1, empty_p4_2, empty_p4_3, empty_p4_4, col_p4_2 = st.columns([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
with col_p4_1:
    st.subheader('* 회사채 동향')
with empty_p4_1:
    st.empty()
with empty_p4_2:
    st.empty()
with empty_p4_3:
    st.empty()
with empty_p4_4:
    st.empty()
with col_p4_2:
    skipped_button_company = st.checkbox('기업 중복기사 생략', key='company_skip',
                                         help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. \n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.")

''
empty_t5, col_t3, empty_t6 = st.columns([0.1, 0.6, 0.1])
with empty_t5:
    st.empty()
with col_t3:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        st.write(':red[▶  전일 대비 조회]')

        today_com = df[(df.수집일자 == crawling_date) & (df.토픽 == '회사채')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today_com.columns = ['제목내최초기업명', 'today_cnt']
        yester_com = df[(df.크롤링일자 == yester_date) & (df.토픽 == 'PE')][
            ['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester_com.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day_com = pd.merge(today_com, yester_com, on='제목내최초기업명', how='left').fillna(0)
        company_2day_com['delta_cnt'] = company_2day_com.today_cnt - company_2day_com.yester_cnt

        # 예외 처리
        today_shape = company_2day_com.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                '제목내최초기업명': ['ㅡ'] * make_null_cnt,
                'today_cnt': [0] * make_null_cnt,
                'yester_cnt': [0] * make_null_cnt,
                'delta_cnt': [0] * make_null_cnt})

            company_2day_com = pd.concat([company_2day_com, null_df]).reset_index(drop=True)


        col_title_top1_com, col_title_top2_com, col_title_top3_com, col_title_top4_com, col_title_top5_com = st.columns(
            5)
        with col_title_top1_com:
            st.metric(label=company_2day_com['제목내최초기업명'][0], value=company_2day_com['today_cnt'][0],
                      delta=company_2day_com['delta_cnt'][0])
        with col_title_top2_com:
            st.metric(label=company_2day_com['제목내최초기업명'][1], value=company_2day_com['today_cnt'][1],
                      delta=company_2day_com['delta_cnt'][1])
        with col_title_top3_com:
            st.metric(label=company_2day_com['제목내최초기업명'][2], value=company_2day_com['today_cnt'][2],
                      delta=company_2day_com['delta_cnt'][2])
        with col_title_top4_com:
            st.metric(label=company_2day_com['제목내최초기업명'][3], value=company_2day_com['today_cnt'][3],
                      delta=company_2day_com['delta_cnt'][3])
        with col_title_top5_com:
            st.metric(label=company_2day_com['제목내최초기업명'][4], value=company_2day_com['today_cnt'][4],
                      delta=company_2day_com['delta_cnt'][4])

    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')

        ### 7day 수정 확인
        sort_com_1 = df[df.토픽 == '회사채'][['제목내최초기업명', '크롤링일자']].dropna(axis=0).groupby(['제목내최초기업명']).max().reset_index()
        sort_com_2 = pd.merge(sort_com_1, df[df.토픽 == '회사채'][['제목내최초기업명', '크롤링일자', '기업기준동일기사개수_일주일']].dropna(axis=0).drop_duplicates(), on=['제목내최초기업명', '크롤링일자'], how='left')
        company_7day_com = sort_com_2.sort_values(by='기업기준동일기사개수_일주일', ascending=False).reset_index(drop=True).drop(['크롤링일자'], axis=1)



        company_7day_com.columns = ['key_', 'cnt']

        # 예외 처리
        today_shape = company_7day_com.shape[0]
        make_null_cnt = 5 - today_shape

        if today_shape < 5:
            null_df = pd.DataFrame({
                'key_': ['ㅡ'] * make_null_cnt,
                'cnt': [0] * make_null_cnt})

            company_7day_com = pd.concat([company_7day_com, null_df]).reset_index(drop=True)

        col_title_top1_com_7, col_title_top2_com_7, col_title_top3_com_7, col_title_top4_com_7, col_title_top5_com_7 = st.columns(
            5)
        with col_title_top1_com_7:
            st.metric(label=company_7day_com['key_'][0], value=company_7day_com['cnt'][0], delta='')
        with col_title_top2_com_7:
            st.metric(label=company_7day_com['key_'][1], value=company_7day_com['cnt'][1], delta='')
        with col_title_top3_com_7:
            st.metric(label=company_7day_com['key_'][2], value=company_7day_com['cnt'][2], delta='')
        with col_title_top4_com_7:
            st.metric(label=company_7day_com['key_'][3], value=company_7day_com['cnt'][3], delta='')
        with col_title_top5_com_7:
            st.metric(label=company_7day_com['key_'][4], value=company_7day_com['cnt'][4], delta='')
with empty_t6:
    st.empty()

''

tmp_company_name_ipo = df[df.수집일자 == crawling_date][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().sort_values(by = '기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
company_name_ipo = list(tmp_company_name_ipo.제목내최초기업명)
radio_sel3_3 = st.multiselect(f"수집 뉴스 조회", company_search_list, default=company_search_list, key='part2_3',
                              max_selections=len(company_search_list))
if skipped_button_company:

    true_type_3['뉴스URL'] = true_type_3['뉴스URL'].apply(make_clickable)
    tmp4_1 = true_type_3[(true_type_3['검색어'].isin(radio_sel3_3))].reset_index(drop=True)#.to_html(escape= False) # render_links=True,
    tmp4_1.columns = ['검색어', '기업명', '타이틀', '발행일시','뉴스URL']
    tmp4_1 = tmp4_1.to_html(escape=False)
    st.write(tmp4_1, unsafe_allow_html=True)

else:
    COM_df['뉴스URL'] = COM_df['뉴스URL'].apply(make_clickable)
    tmp4_2 = COM_df[(COM_df['검색어'].isin(radio_sel3_3))].reset_index(drop=True)#.to_html(escape=False)
    tmp4_2.columns = ['검색어', '기업명', '타이틀', '발행일시', '뉴스URL']
    tmp4_2 = tmp4_2.to_html(escape=False)
    st.write(tmp4_2, unsafe_allow_html=True)


'-----'
