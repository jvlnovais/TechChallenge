import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


st.set_page_config(layout="wide")
st.title("Dashboard Interativo - Preço em US$ e Consumo em TWh Petróleo")

df = pd.read_excel('preco_petroleo.xlsx')
df.columns = ['data', 'preco']
df['data'] = pd.to_datetime(df['data'])
df = df.dropna().sort_values('data')

de = pd.read_excel('global_energy_usage.xlsx')
de.columns = ['data','outros renovaveis','bio-combustivel','energia solar','energia eolica','hidroeletricas','nuclear','gas','petroleo','carvao','biomassa']
de['data'] = pd.to_datetime(de['data'], format='%Y')
de = de.dropna().sort_values('data')

fig = px.line(df, x='data', y='preco', title='Histórico do Preço')
st.plotly_chart(fig, use_container_width=True)

fig = px.line(de, x='data', y='petroleo', title='Uso de Petróleo')
st.plotly_chart(fig, use_container_width=True)

df_prophet = df.rename(columns={'data': 'ds', 'preco': 'y'})
modelo = Prophet(daily_seasonality=True)
modelo.fit(df_prophet)

dias = st.slider("Quantos dias deseja prever?", 7, 90, 30)
futuro = modelo.make_future_dataframe(periods=dias)
previsao = modelo.predict(futuro)

fig2 = px.line(previsao, x='ds', y='yhat', title='Previsão de Preço para o Brent')
st.plotly_chart(fig2, use_container_width=True)

st.title("Eventos Importantes para a História do Petróleo")

st.subheader("📉 Impacto de Eventos Históricos no Preço do Petróleo")

eventos = {
    "Crise de 2008": "2008-09-15",
    "COVID-19": "2020-03-11",
    "Guerra na Ucrânia": "2022-02-24",
    "Corte da OPEP+": "2023-04-01",
    "Guerra do Golfo": "1990-08-02",
    "Ataque em Abqaiq-Khurais": "2019-09-14"
}

for nome, data in eventos.items():
    data_evento = pd.to_datetime(data)
    inicio = data_evento - pd.DateOffset(months=12)
    fim = data_evento + pd.DateOffset(months=12)
    
    df_evento = df[(df['data'] >= inicio) & (df['data'] <= fim)]

    fig = go.Figure()

   
    fig.add_trace(go.Scatter(
        x=df_evento['data'], y=df_evento['preco'],
        mode='lines',
        name='Preço Brent',
        line=dict(color='royalblue', width=2)
    ))


    fig.add_vline(
        x=data_evento,
        line_width=2,
        line_dash="dash",
        line_color="red"
    )


    fig.add_annotation(
        x=data_evento,
        y=max(df_evento['preco']),
        text=nome,
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(color="red", size=12),
        bgcolor="white"
    )

    fig.update_layout(
        title=f"📌 Evento: {nome}",
        xaxis_title="Data",
        yaxis_title="Preço do Petróleo (USD)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
