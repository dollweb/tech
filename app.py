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

url = 'https://raw.githubusercontent.com/dollweb/tech/main/창업기업평가_하나은행_2022_원본_10_10개독립변수.csv'
encoded_url = urllib.parse.quote(url, safe=':/')
df = pd.read_csv(encoded_url)
y = df.T_SC
tf.compat.v1.disable_eager_execution()
tf.compat.v1.global_variables_initializer()
model_tf = tf.compat.v1.global_variables_initializer()
X_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 16])
y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([16, 1]), name="weight") # 가중치 초기화
b = tf.Variable(tf.random.normal([1]), name="bias") # 바이어스값 초기화
hypothesis = tf.matmul(X_tf, W) + b
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

col1, col2 = st.columns(2)
with col1:
   st.subheader("기술정보입력")
   with st.form("form1"):
      T_14_temp = st.text_input('자본조달능력: ')
      T_19_temp = st.text_input('모방의난이도: ')
      T_24_temp = st.text_input('시장성장성: ')
      T_26_temp = st.text_input('인지도: ')
      T_28_temp = st.text_input('경쟁제품과비교우위성: ')
      T_32_temp = st.text_input('종업원수: ')
      T_39_temp = st.text_input('동업종종사기간: ')
      T_43_temp = st.text_input('연구개발조직운영기간: ')
      T_47_temp = st.text_input('인증누적건수: ')
      T_79_temp = st.text_input('업력: ')
      T_80_temp = st.text_input('벤처+이노비즈: ')
      T_82_temp = st.text_input('지식재산권전체합계: ')
      T_83_temp = st.text_input('매출액2년평균: ')
      T_101_temp = st.text_input('연구개발비2년평균: ')
      T_111_temp = st.text_input('연구소디자인더미: ')
      T_115_temp = st.text_input('3개년개발비더미: ')

      submit = st.form_submit_button("확인")
      등급 = ""
      if submit:
         with tf.compat.v1.Session() as sess:
            sess.run(model_tf)
            
            save_path = "./saved.cpkt"
            url = 'https://github.com/dollweb/tech/raw/main/saved.cpkt'
            response = requests.get(url)
            with open(save_path, "wb") as f:
               f.write(response.content)
            saver.restore(sess, save_path)
            scaler_T_SC = MinMaxScaler()
            input_data = np.array([[T_14_temp, T_19_temp, T_24_temp, T_26_temp, T_28_temp, T_32_temp, T_39_temp, T_43_temp, T_47_temp, T_79_temp, T_80_temp, T_82_temp, T_83_temp, T_101_temp, T_111_temp, T_115_temp]]).reshape(1, -1)
            input_data_normalized = scaler_T_SC.fit_transform(input_data)
            arr = np.array(input_data_normalized, dtype=np.float32)
            x_data = arr[0:16]
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