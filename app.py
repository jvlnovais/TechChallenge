import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


st.set_page_config(layout="wide")
st.title("Dashboard Interativo - Preço Petróleo")

df = pd.read_excel('preco_petroleo.xlsx')
df.columns = ['data', 'preco']
df['data'] = pd.to_datetime(df['data'])
df = df.dropna().sort_values('data')

fig = px.line(df, x='data', y='preco', title='Histórico do Preço')
st.plotly_chart(fig, use_container_width=True)

df_prophet = df.rename(columns={'data': 'ds', 'preco': 'y'})
modelo = Prophet(daily_seasonality=True)
modelo.fit(df_prophet)

dias = st.slider('Quantos dias deseja prever?', 7, 90, 30)
futuro = modelo.make_future_dataframe(periods=dias)
previsao = modelo.predict(futuro)

fig2 = px.line(previsao, x='ds', y='yhat', title='Previsão de Preço para o Brent')
st.plotly_chart(fig2, use_container_width=True)
