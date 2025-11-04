import streamlit as st
import numpy as np
import pandas as pd


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df_train = load_data("data/train.csv")


st.title("Hackathon Heart Attack")

st.dataframe(df_train.head())

button = st.button("Appuie ici")

if button :

    st.write("Caca")


