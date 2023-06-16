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
    with open('./resources/news_lottie2.json', 'r', encoding='utf-8-sig') as st_json:
        return json.load(st_json)

# # test about link
# def make_clickable(link):
#     text = link.split('=')[1]
#     return f'<a target="_blank" href="{link}">{text}</a>'


# wordcloud 함수
# def plotChart(count_dict, max_words_, container):
#     # 백그라운드 마스크 이미지.
#     img = Image.open('./resources/background_0.png')
#     my_mask = np.array(img)
#     # 워드 클라우드 객체.
#     wc = WordCloud(font_path='./resources/NanumSquareRoundEB.ttf',
#                    background_color='black',
#                    contour_color='grey',
#                    width = 1000,
#                    contour_width=0.01,
#                    max_words=max_words_,
#                    mask=my_mask)
#     wc.generate_from_frequencies(count_dict)
#     fig = plt.figure(figsize=(10, 3))
#     plt.imshow(wc, interpolation='bilinear')
#     plt.axis("off")
#     container.pyplot(fig)

def return_key_list(texts):
    return texts.replace(' ', '').replace('[', '').replace(']','').replace('"','').split(',')



################ 환경 설정 ################




# DATA LOAD
df = loadPickle('./resources/all_full_db.pkl')         #all_full_db_060708


#제목 일자 설정
df['수집일자'] = df['크롤링일자'].dt.strftime('%Y-%m-%d')
crawling_date = df['수집일자'].max()
yester_date = df.크롤링일자.max() - pd.Timedelta(1, unit='days')


# 요약키워드 리스트만들기
df['요약키워드리스트'] = df.요약키워드.apply(return_key_list)
all_keyword = []
for i in df.요약키워드리스트:
    all_keyword += i
key_dict = collections.Counter(all_keyword)





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
    ''
    st.title('뉴스 데이터 한눈에 보기')
    st.write(':red[배치ver (매일 X시 기준)]')

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
''
''
st.write("---")
st.write(f'{crawling_date}일 기준 수집된 전체 주요 기사는 {df.shape[0]}개 입니다.')
''
''


# st.subheader('* 워드클라우드', help= "뉴스 전문의 검색어를 추출하여 워드클라우드로 출력하였습니다.")
# empty_image_1, image_set, empty_image_2 = st.columns([0.3, 0.7, 0.3])
# with empty_image_1:
#     st.empty()
# with image_set:
#     wordcloud_st = st.empty()
#     plotChart(key_dict, 80, wordcloud_st)
# with empty_image_2 :
#     st.empty()
# ''



st.subheader('* 검색어 TOP 5', help="당일 수집된 검색어별 기사 수를 집계한 결과이며,\n 전일자의 기사수와 어느 정도 차이나는지 확인할 수 있습니다. ")

# dataframe 만들기
keyword_today = df[df.수집일자 == crawling_date].검색어.value_counts().reset_index()
keyword_today.columns = ['key_', 't_cnt']
keyword_yesterday = df[df.크롤링일자 == yester_date].검색어.value_counts().reset_index()
keyword_yesterday.columns = ['key_', 'y_cnt']
top_today = pd.merge(keyword_today, keyword_yesterday, on ='key_', how='left').fillna(0)
top_today['delta_cnt'] = top_today.t_cnt - top_today.y_cnt
# top_today = top_today.astype({'delta_cnt':'int'})



# top_yesterday해서 가져오기 → delta 값 수정
empty_p1_1, col_p1_1, col_p1_2, col_p1_3, col_p1_4, col_p1_5, empty_p1_2 = st.columns([0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2])
with empty_p1_1:
    st.empty()

with col_p1_1:
    st.metric(label=top_today['key_'][0], value=top_today['t_cnt'][0], delta=top_today['delta_cnt'][0],  delta_color="inverse")
with col_p1_2:
    st.metric(label=top_today['key_'][1], value=top_today['t_cnt'][1], delta=top_today['delta_cnt'][1],  delta_color="inverse")
with col_p1_3:
    st.metric(label=top_today['key_'][2], value=top_today['t_cnt'][2], delta=top_today['delta_cnt'][2],  delta_color="inverse")
with col_p1_4:
    st.metric(label=top_today['key_'][3], value=top_today['t_cnt'][3], delta=top_today['delta_cnt'][3],  delta_color="inverse")
with col_p1_5:
    st.metric(label=top_today['key_'][4], value=top_today['t_cnt'][4], delta=top_today['delta_cnt'][4],  delta_color="inverse")

with empty_p1_2:
    st.empty()

''
'---'
''

##### IPO PART
len_2_1 = df[df['검색어'].isin(['상장연기', '프리 IPO', '상장예비심사', 'IPO 주관'])].shape[0]

# 중복제거
true_type_1 = df[(df.중복기업기사제거 == False) & (df['검색어'].isin(['상장연기', '프리 IPO', '상장예비심사', 'IPO 주관']))][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']].reset_index(drop=True)

# 중복제거x
IPO_df = df[(df.토픽 == 'IPO')][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']]
# col_p2_1, empty_p2_4, col_p2_2 = st.columns([0.6, 0.1, 0.2])
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
    skipped_button_IPO = st.checkbox('기업 중복기사 생략', help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. "+
                                                        "\n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.")



##### css 적용한 기업top
''
empty_t3, col_t2, empty_t2 = st.columns([0.1, 0.6, 0.1])
with empty_t3:
    st.empty()
    ''
with col_t2:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        ''
        st.write(':red[▶  전일 대비 조회]')

        today = df[(df.수집일자 == crawling_date) & (df.토픽 == 'IPO')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today.columns = ['제목내최초기업명', 'today_cnt']
        yester = df[(df.크롤링일자 == yester_date)& (df.토픽 == 'IPO')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day = pd.merge(today, yester, on='제목내최초기업명', how='left').fillna(0)
        company_2day['delta_cnt'] = company_2day.today_cnt - company_2day.yester_cnt

        # company_2day = df[df.수집일자 == crawling_date][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(axis = 0).sort_values(by = '기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        # company_2day.columns = ['key_', 'cnt']

        col_lottie1, col_t2itle2, col33, col44, col55 = st.columns(5)
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
    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')

        company_7day = df[['제목내최초기업명', '기업기준동일기사개수_일주일']].drop_duplicates().sort_values(by='기업기준동일기사개수_일주일',
                                                                                        ascending=False).reset_index(
            drop=True)
        company_7day.columns = ['key_', 'cnt']

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
''
''

radio_sel1_3 = st.multiselect(f"수집 뉴스 조회", ['상장연기', '프리 IPO', '상장예비심사', 'IPO 주관']
                              , default=['상장연기', '프리 IPO', '상장예비심사', 'IPO 주관'], key='part2_1', max_selections=4
                              , help=f"전체 기사 수 : {len_2_1}   →   중복기업기사 제거 경우의 기사수 : {true_type_1.shape[0]}이며, 각 열별로 원하시는 정렬이 가능합니다.")

if skipped_button_IPO :
    st.dataframe(true_type_1[true_type_1['검색어'].isin(radio_sel1_3)].reset_index(drop=True), 200000, 500, use_container_width=True)
else :
    st.dataframe(IPO_df[IPO_df['검색어'].isin(radio_sel1_3)].reset_index(drop=True), 200000, 500, use_container_width=True)



# 기존
# st.dataframe(df[df['검색어'].isin(radio_sel1_3)].reset_index(drop=True), 200000, 500, use_container_width=True)
'-----'










##### PE PART
len_2_2 = df[df['검색어'].isin(['M&A', '경영권 인수', '공개매수'])].shape[0]
true_type_2 = df[(df.중복기업기사제거 == False) &(df['토픽'] == 'PE')][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']].reset_index(drop=True)
PE_df = df[(df.토픽 == 'PE')][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']]

col_p3_1, empty_p3_1, empty_p3_2, empty_p3_3, empty_p3_4, col_p3_2= st.columns([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
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
    skipped_button_PE = st.checkbox('기업 중복기사 생략', help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. \n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.", key='PE_BUTTON')
    ''


empty_t3, col_t2, empty_t4 = st.columns([0.1, 0.6, 0.1])
with empty_t3:
    st.empty()
    ''
with col_t2:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        ''
        st.write(':red[▶  전일 대비 조회]')

        today_PE = df[(df.수집일자 == crawling_date) & (df.토픽 == 'PE')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today_PE.columns = ['제목내최초기업명', 'today_cnt']
        yester_PE = df[(df.크롤링일자 == yester_date)& (df.토픽 == 'PE')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester_PE.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day_PE = pd.merge(today_PE, yester_PE, on='제목내최초기업명', how='left').fillna(0)
        company_2day_PE['delta_cnt'] = company_2day_PE.today_cnt - company_2day_PE.yester_cnt

        # company_2day = df[df.수집일자 == crawling_date][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(axis = 0).sort_values(by = '기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        # company_2day.columns = ['key_', 'cnt']

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
    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')

        company_7day_PE = df[df.토픽 == 'PE'][['제목내최초기업명', '기업기준동일기사개수_일주일']].drop_duplicates().sort_values(by='기업기준동일기사개수_일주일',
                                                                                        ascending=False).reset_index(
            drop=True)
        company_7day_PE.columns = ['key_', 'cnt']

        col_title_top1_pe_7, col_title_top2_pe_7, col_title_top3_pe_7, col_title_top4_pe_7, col_title_top5_pe_7 = st.columns(5)
        with col_title_top1_pe_7:
            st.metric(label=company_7day_PE['key_'][0], value=company_7day_PE['cnt'][0], delta='')
        with col_title_top2_pe_7:
            st.metric(label=company_7day_PE['key_'][1], value=company_7day['cnt'][1], delta='')
        with col_title_top3_pe_7:
            st.metric(label=company_7day_PE['key_'][2], value=company_7day_PE['cnt'][2], delta='')
        with col_title_top4_pe_7:
            st.metric(label=company_7day_PE['key_'][3], value=company_7day_PE['cnt'][3], delta='')
        with col_title_top5_pe_7:
            st.metric(label=company_7day_PE['key_'][4], value=company_7day_PE['cnt'][4], delta='')
with empty_t4:
    st.empty()
''
''
''
radio_sel2_3 = st.multiselect(f"수집 뉴스 조회", ['M&A', '경영권 인수', '공개매수']
                              , default=['M&A', '경영권 인수', '공개매수'], key='part2_2', max_selections=3, help=f"검색어 선택 전체 기사 수 : {len_2_2}   →   중복기업기사 제거경우의 기사수 : {true_type_2.shape[0]}이며, 각 열별로 원하시는 정렬이 가능합니다.")

if skipped_button_PE :
    st.dataframe(true_type_2[true_type_2['검색어'].isin(radio_sel2_3)].reset_index(drop=True), 200000, 500, use_container_width=True)
else :
    st.dataframe(PE_df[PE_df['검색어'].isin(radio_sel2_3)].reset_index(drop=True), 200000, 500, use_container_width=True)


# st.dataframe(df[df['검색어'].isin(radio_sel2_3)].reset_index(drop=True), 200000, 500, use_container_width=True)

'-----'



##### 회사채 PART

len_2_3 = df[df['검색어'].isin(['유상증자', '회사채 발행', '회사채 공모', '회사채 수요예측'])].shape[0]

# 중복제거 VER
true_type_3 = df[(df.중복기업기사제거 == False) & (df['검색어'].isin(['유상증자', '회사채 발행', '회사채 공모', '회사채 수요예측']))][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']].reset_index(drop=True)

# 중복제거X
COM_df = df[(df.토픽 == '회사채')][['검색어','발행일시','언론사','타이틀','요약키워드리스트','뉴스URL','지피티스코어','제목스코어','인덱스']]

col_p4_1, empty_p4_1, empty_p4_2, empty_p4_3, empty_p4_4, col_p4_2= st.columns([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
with col_p4_1:
    st.subheader('* 회사채 동향')
with empty_p4_1 :
    st.empty()
with empty_p4_2 :
    st.empty()
with empty_p4_3 :
    st.empty()
with empty_p4_4 :
    st.empty()
with col_p4_2 :
    skipped_button_company = st.checkbox('기업 중복기사 생략', key='company_skip', help="체크 박스 선택을 통해 동일 기업에 대한 기사를 중복기사로 판단하여 생략합니다. \n 제거 후 남은 기사는 다른 기사에 비해 유의미하다고 선별된 기사입니다.")







''
empty_t5, col_t3, empty_t6 = st.columns([0.1, 0.6, 0.1])
with empty_t5:
    st.empty()
    ''
with col_t3:
    with st.container():
        st.subheader('* 기업 TOP 5', help="해당 기업에 대한 기사 수를 전일자 및 일주일 전에 대비 하여 집계 조회한 결과 입니다.")
        ''
        st.write(':red[▶  전일 대비 조회]')

        today_com = df[(df.수집일자 == crawling_date) & (df.토픽 == '회사채')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        today_com.columns = ['제목내최초기업명', 'today_cnt']
        yester_com = df[(df.크롤링일자 == yester_date)& (df.토픽 == 'PE')][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(
            axis=0).sort_values(by='기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        yester_com.columns = ['제목내최초기업명', 'yester_cnt']
        # 2일치
        company_2day_com = pd.merge(today_com, yester_com, on='제목내최초기업명', how='left').fillna(0)
        company_2day_com['delta_cnt'] = company_2day_com.today_cnt - company_2day_com.yester_cnt

        # company_2day = df[df.수집일자 == crawling_date][['제목내최초기업명', '기업기준동일기사개수_이틀간']].drop_duplicates().dropna(axis = 0).sort_values(by = '기업기준동일기사개수_이틀간', ascending=False).reset_index(drop=True)
        # company_2day.columns = ['key_', 'cnt']

        col_title_top1_com, col_title_top2_com, col_title_top3_com, col_title_top4_com, col_title_top5_com = st.columns(5)
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
    ''

    with st.container():
        st.write(':red[▶  누적 7일 기준]')

        company_7day_com = df[df.토픽 == '회사채'][['제목내최초기업명', '기업기준동일기사개수_일주일']].drop_duplicates().sort_values(by='기업기준동일기사개수_일주일',
                                                                                        ascending=False).reset_index(
            drop=True)
        company_7day_com.columns = ['key_', 'cnt']

        col_title_top1_com_7, col_title_top2_com_7, col_title_top3_com_7, col_title_top4_com_7, col_title_top5_com_7 = st.columns(5)
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
''
''









radio_sel3_3 = st.multiselect(f"수집 뉴스 조회", ['유상증자', '회사채 발행', '회사채 공모', '회사채 수요예측']
                              , default=['유상증자', '회사채 발행', '회사채 공모', '회사채 수요예측'], key='part2_3', max_selections=5, help=f"전체 기사 수 : {len_2_3}   →   중복기업기사 제거경우의 기사수 : {true_type_3.shape[0]}이며, 각 열별로 원하시는 정렬이 가능합니다.")

if skipped_button_company :

    st.dataframe(true_type_3[true_type_3['검색어'].isin(radio_sel3_3)].reset_index(drop=True), 200000, 500, use_container_width=True)

else :
    st.dataframe(COM_df[COM_df['검색어'].isin(radio_sel3_3)].reset_index(drop=True), 200000, 500, use_container_width=True)
# st.markdown(df[df['검색어'].isin(radio_sel3_3)].reset_index(drop=True).to_html(render_links=True), unsafe_allow_html=True)
'-----'

