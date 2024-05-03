import streamlit as st
import warnings
import plotly.express as px
import pandas as pd
import os
import seaborn as sns

warnings.filterwarnings('ignore')

#Page Setup
st.set_page_config(page_title="Dashboard", page_icon=':chart_with_upwards_trend:')
st.title(":chart_with_upwards_trend: DashBoard")
st.markdown("<style> div.block-container{padding-top:1rem;} </style>",unsafe_allow_html=True)

#Reading File
df = pd.read_csv("C:/Users/AYUSHI/OneDrive/Desktop/processed_file.csv")
st.write(df.head())

# Year Filter 

col1,col2 = st.columns((2))

startYear = df["Year"].min()
EndYear = df["Year"].max()
with col1:
    Year1 = int(st.number_input("Start Year",min_value= startYear, max_value= EndYear))

with col2:
    Year2 = int(st.number_input("End Year", max_value=EndYear,placeholder=EndYear, min_value=startYear))

df = df[(df['Year']>=Year1 ) & (df["Year"]<=Year2)].copy()

# Country Filter
st.sidebar.header("Choose your Filter: ")
region = st.sidebar.multiselect("Region",df['Country'].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Country"].isin(region)]


#PLOTING
with col1:
    st.subheader("Data: ")
    st.write(df2)

with col1:
    st.subheader("Yearwise Emissions")
    for i in ['NO', 'SO2', 'CO', 'OC', 'NMVOCs', 'BC', 'NH3']:
        fig = px.box(df2, x = "Year", y = i, template="seaborn", color="Year")
        st.plotly_chart(fig, height = 300)

with col1:
    st.subheader("Distributions ")
    for i in ['NO', 'SO2', 'CO', 'OC', 'NMVOCs', 'BC', 'NH3']:
        fig = px.histogram(df2, x = i, template="seaborn")
        st.plotly_chart(fig, height = 300)

with col1:
    st.subheader("HeatMap of Pollutants and Diseases")
    columns_of_interest = ['NO', 'SO2', 'CO', 'OC', 'NMVOCs', 'BC', 'NH3','D_LRI', 'D_CRD', 'D_TBLC']
    df2.loc[:, columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')
    corr_matrix = df2[columns_of_interest].corr()
    fig = px.imshow(corr_matrix,text_auto=True)
    st.plotly_chart(fig)

# Error Plotting
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

predictors = df[['D_LRI', 'D_CRD', 'D_TBLC']]
x = df[['NO', 'SO2', 'CO', 'OC', 'NMVOCs', 'BC', 'NH3']]
x_train,x_test,y_train,y_test  = train_test_split(x,predictors, test_size=0.25, random_state=42)
model = LinearRegression().fit(x_train,y_train)
y_predict = model.predict(x_test)

with col1:
    st.header("Linear Regression")
    st.subheader("R2 Score")
    st.write(metrics.r2_score(y_test,y_predict))
    st.subheader("Root Mean squared Error:")
    st.write(metrics.mean_squared_error(y_test,y_predict))


from sklearn.ensemble import RandomForestRegressor as rfr
Rfr_model = rfr().fit(x_train,y_train)
rfr_predict = Rfr_model.predict(x_test)
with col1:
    st.header("Random Forest Regression")
    st.subheader("R2 Score")
    st.write(metrics.r2_score(y_test,rfr_predict))
    st.subheader("Root Mean squared Error:")
    st.write(metrics.mean_squared_error(y_test,rfr_predict))

st.header("From the above data it is evident that the relationship between the emmitted gases and cancer is NOT LINEAR.")



