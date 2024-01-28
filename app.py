# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import urllib.parse
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import subprocess
import requests
import io

def install_requirements():
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
install_requirements()

st.set_page_config(
    page_title="TECH"
)

st.title("TECH!")
st.subheader("기술정보를 입력해주세요")
st.markdown("**기술정보**를 입력해주세요")
st.text("기술정보를 입력해주세요")

url = 'https://raw.githubusercontent.com/dollweb/tech/main/창업기업평가_하나은행_2022_원본_05_정규화.csv'
encoded_url = urllib.parse.quote(url, safe=':/')
df = pd.read_csv(encoded_url)
# url = 'https://raw.githubusercontent.com/dollweb/tech/main/창업기업평가_하나은행_2022_원본_05_정규화.csv'
# df = pd.read_csv(url, encoding='utf-8')
y = df.T_SC
tf.compat.v1.disable_eager_execution()
tf.compat.v1.global_variables_initializer()
model_tf = tf.compat.v1.global_variables_initializer()
X_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 17])
y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([17, 1]), name="weight") # 가중치 초기화
b = tf.Variable(tf.random.normal([1]), name="bias") # 바이어스값 초기화
hypothesis = tf.matmul(X_tf, W) + b
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

col1, col2 = st.columns(2)
with col1:
   st.subheader("기술정보입력")
   with st.form("form1"):
      T_32_temp = st.text_input(label="종업원수")
      T_37_temp = st.text_input(label="이노비즈인증유무")
      T_43_temp = st.text_input(label="연구개발조직운영기간")
      T_47_temp = st.text_input(label="인증누적건수")
      T_50_temp = st.text_input(label="특허등록건수")
      T_53_temp = st.text_input(label="상표권등록건수")
      T_55_temp = st.text_input(label="프로그램등록건수")
      T_56_temp = st.text_input(label="평가기업기준년도매출액")
      T_57_temp = st.text_input(label="평가기업전년도매출액")
      T_63_temp = st.text_input(label="업종평균전년도매출액영업이익률")
      T_67_temp = st.text_input(label="평가기업전전년도연구개발비")
      T_80_temp = st.text_input(label="벤처+이노비즈")
      T_82_temp = st.text_input(label="지식재산권전체합계")
      T_83_temp = st.text_input(label="매출액2년평균")
      T_99_temp = st.text_input(label="연구개발비3년평균")
      T_106_temp = st.text_input(label="연구소디자인더미")
      T_110_temp = st.text_input(label="3개년개발비더미")

      submit = st.form_submit_button("확인")
      등급 = ""
      if submit:
         with tf.compat.v1.Session() as sess:
            sess.run(model_tf)
            # save_path = 'C:/research/saved.cpkt'
            # save_path = '/content/drive/MyDrive/research/saved.cpkt'
            # save_path = 'https://raw.githubusercontent.com/dollweb/tech/main/saved.cpkt'
            
            save_path = 'https://github.com/dollweb/tech/raw/main/saved.cpkt'
            response = requests.get(save_path)
            content = response.content
            memory_file = io.BytesIO(content)
            # content_str = content.decode('utf-8')
            # memory_file = io.BytesIO(content_str.encode('utf-8'))
            saver.restore(sess, memory_file)
            
            # save_path = 'https://raw.githubusercontent.com/dollweb/tech/main/'
            # save_path = 'https://raw.githubusercontent.com/dollweb/tech/main/saved.cpkt'
            # save_path = "https://raw.githubusercontent.com/dollweb/tech/main/saved.cpkt"
            # save_path = 'https://raw.githubusercontent.com/dollweb/tech/main/checkpoint.cpkt'
            # saver.restore(sess, save_path)

            scaler_T_SC = MinMaxScaler()
            input_data = np.array([[T_32_temp, T_37_temp, T_43_temp, T_47_temp, T_50_temp, T_53_temp, T_55_temp, T_56_temp, T_57_temp, T_63_temp, T_67_temp, T_80_temp, T_82_temp, T_83_temp, T_99_temp, T_106_temp, T_110_temp]]).reshape(1, -1)
            input_data_normalized = scaler_T_SC.fit_transform(input_data)
            arr = np.array(input_data_normalized, dtype=np.float32)
            x_data = arr[0:17]
            prediction = sess.run(hypothesis, feed_dict={X_tf: x_data})
            predicted_T_SC_NM = prediction[0][0]
            print(f"T_SC_NM 예측된 정규화값: {predicted_T_SC_NM}")
            scaler_T_SC.fit(y.values.reshape(-1, 1))
            predicted_T_SC = scaler_T_SC.inverse_transform(prediction)[0][0]
            print(f"T_SC 역정규화 값: {predicted_T_SC}")
          
         if predicted_T_SC < 20:
            등급 = "T10"
         elif predicted_T_SC < 39:
            등급 = "T9"
         elif predicted_T_SC < 44:
            등급 = "T8"
         elif predicted_T_SC < 48:
            등급 = "T7"
         elif predicted_T_SC < 51:
            등급 = "T6-"
         elif predicted_T_SC < 54:
            등급 = "T6"
         elif predicted_T_SC < 57:
            등급 = "T6+"
         elif predicted_T_SC < 60:
            등급 = "T5-"
         elif predicted_T_SC < 64:
            등급 = "T5"
         elif predicted_T_SC < 67:
            등급 = "T5+"
         elif predicted_T_SC < 70:
            등급 = "T4-"
         elif predicted_T_SC < 73:
            등급 = "T4"
         elif predicted_T_SC < 76:
            등급 = "T4+"
         elif predicted_T_SC < 79:
            등급 = "T3-"
         elif predicted_T_SC < 83:
            등급 = "T3"
         elif predicted_T_SC < 86:
            등급 = "T3+"
         elif predicted_T_SC < 92:
            등급 = "T2"
         else:
            등급 = "T1"

with col2:
    st.subheader("기술등급 예측결과")
    with st.form("form2"):
       st.title(f"{등급}")
       st.form_submit_button("restart")