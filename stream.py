import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


st.markdown("""
    <style>
    .title {
        color: #FF0000; /* Red color */
        font-size: 48px; /* Increase font size */
        font-weight: normal; /* Remove bold */
        text-align: center; /* Center align the title */
        margin-bottom: 20px; /* Add space below the title */
    }
    </style>
    <div class="title">
        Demand Forecasting
    </div>
""", unsafe_allow_html=True)
if 'Home' not in st.session_state:
    st.session_state['Home'] = True
def display(Year,Month,model,Duration,outlet_code):
    st.session_state['Home']=False
    Home=False
    print(Home)
    print(model,Duration,Month,Year,outlet_code)
    print(type(outlet_code))
    Data=pd.read_csv('my_file.csv')
    valid_outlet=list(set(Data['outlet_code'].tolist()))
    if outlet_code!=None:
        try:
            outlet_code=float(outlet_code)
        except ValueError:
            outlet_code=None

    if (outlet_code==None) and model!=None and Duration!=None and Year<=datetime.now().year and Month<=7:
        if 'error' in st.session_state:
            del st.session_state['error']
        Data=Data[(Data['month']<Month+Duration)&(Data['month']>=Month)]
        st.session_state['result']='over all average required'
        if model=='Prophet':
            print('entered prophet')
            final=Data.groupby(['month','year']).agg({'prophet_pred':'sum','yact':'sum'}).reset_index()
            final['yact']=final['yact'].apply(lambda x:round(x,0))
            final['prophet_pred']=final['prophet_pred'].apply(lambda x:round(x,0))
            final['year']=final['year'].apply(lambda x:(x))
            final['month']=final['month']
            print(type(final['month'][0]))
            st.session_state['final_table']=final
            
        elif model=='Random Forest':
            final=Data.groupby(['month','year']).agg({'Random_forest_pred':'sum','yact':'sum'}).reset_index()
            final['yact']=final['yact'].apply(lambda x:round(x,0))
            final['Random_forest_pred']=final['Random_forest_pred'].apply(lambda x:round(x,0))
            final['year']=final['year'].apply(lambda x:(str(x)))
            final['month']=final['month']
            st.session_state['final_table']=final
        elif model=='ARIMA':
                    final=Data.groupby(['month','year']).agg({'Arima_pred':'sum','yact':'sum'}).reset_index()
                    st.session_state['final_table']=final
    
    elif outlet_code  in valid_outlet and model!=None and Year<=datetime.now().year and Month<7 :
        Data=Data[(Data['month']<=Month+Duration)&(Data['outlet_code']==outlet_code)&(Data['month']>=Month)]
        st.session_state['result']='over all average required'
        if model=='Prophet':
            print('entered prophet')
            final=Data.groupby(['month','year']).agg({'prophet_pred':'sum','yact':'sum'}).reset_index()
            st.session_state['final_table']=final
            print(final)
        elif model=='Random Forest':
            final=Data.groupby(['month','year']).agg({'Random_forest_pred':'sum','yact':'sum'}).reset_index()
            st.session_state['final_table']=final
            print(final)
        elif model=='ARIMA':
            final=Data.groupby(['month','year']).agg({'Arima_pred':'sum','yact':'sum'}).reset_index()
            st.session_state['final_table']=final
            print(final)
    else:
        st.session_state['error']='error occured with input'
        if 'final_table' in st.session_state:
            del st.session_state['final_table']
 
st.sidebar.image('MM LOGO.png',width=100)
 
 
st.sidebar.header('Input Parameters')
 

# Model selection\
Date=st.sidebar.date_input('select',value='today')
model_name = st.sidebar.selectbox(
    'Select Forecast Model',
    ['Random Forest', 'Prophet','ARIMA']
)
 
# Forecast duration
duration = st.sidebar.selectbox(
    'Select Forecast Duration (Months)',
    [1,3,6]
)
 
# Outlet code
outlet_code = st.sidebar.text_input('Enter Outlet Code', placeholder='000',value=None)
 
st.sidebar.markdown(f'**Selected Outlet Code**: {outlet_code}')
st.sidebar.button(label='Predict',on_click=display,args=(
int(datetime.strftime(Date,'%Y')),int(datetime.strftime(Date,'%m')),model_name,duration,outlet_code),  type='primary')

if 'error' in st.session_state:
    st.write(':red[{}]'.format(st.session_state['error']))
if 'final_table' in st.session_state:
    outlet='all'
    
    if outlet_code!=None:
        if outlet_code!=None:
            try:
                outlet_code=float(outlet_code)
            except ValueError:
                outlet_code=None
        out=pd.read_csv('outlet_data.csv')
        if(outlet_code!=None):
            out=out[out['outlet_code_dms']==float(outlet_code)]
            outlet=out['outlet_name'].tolist()
            outlet=outlet[0]
    if outlet_code!=None:
        outlets='outlet'
    else:
        outlets='outlets'
    st.header('Selected Model is {} and for {} {}'.format(model_name,outlet,outlets))
    start,center,end=st.columns([1,5,2])
    with center:
        st.dataframe(st.session_state['final_table'])
    with end:
        Da=pd.read_csv('final_results.csv')
        Da['Model']=['Random Forest','ARIMA','Prophet']
        Da=Da[Da['Model']==model_name]
        r2score=Da['R2_square'].tolist()
        r2score=round(r2score[0],2)
        MSE=Da['MSE'].tolist()
        MSE=round(MSE[0],2)

        co=1
        lo=2
        st.markdown(f"""
    <style>
    .container {{
        padding-left: 10px;
        padding-right:10px;
        flex-direction: column;
        justify-content: center;
        height:auto;
        width:fit-content; 
        border-style:solid;
        border-width:2px   }}
    p{{
                    color:red;
    }}
    </style>
    <div class='container'>
        <h4>Metrics</h4>
        <p>R2_score: <strong>{r2score}</strong></p>
        <p>MSE: <strong>{MSE}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    if model_name=='Prophet':
        # st.header('selected model is : Prophet')
        Da=st.session_state['final_table']
        # print(Da['Actual'])
        act=Da['yact']
        pred=Da['prophet_pred']
        plt.figure(figsize=(10, 5))
        plt.title('Actual VS Predicted')
        plt.plot(Da['month'],Da['yact'],label='actual')
        plt.plot(Da['month'],Da['prophet_pred'],label='predicted')
        plt.xlabel('Month')
        plt.ylabel('Number Of Units')
        plt.legend()
        x_ticks = np.round(Da['month']).astype(int)
        plt.xticks(x_ticks)
        st.pyplot(plt)
        # st.line_chart(data=st.session_state['final_table'],x='month',y=['yact','prophet_pred'])
    elif model_name=='Random Forest':
        Da=st.session_state['final_table']
        # print(Da['Actual'])
        act=Da['yact']
        pred=Da['Random_forest_pred']
        # st.line_chart(data=Da,x='month',y=['yact','Random_forest_pred'])
        plt.figure(figsize=(10, 5))
        plt.title('Actual VS Predicted')
        plt.plot(Da['month'],Da['yact'],label='actual')
        plt.plot(Da['month'],Da['Random_forest_pred'],label='predicted')
        plt.xlabel('Month')
        plt.ylabel('Number Of Units')
        plt.legend()
        x_ticks = np.round(Da['month']).astype(int)
        plt.xticks(x_ticks)
        st.pyplot(plt)
    elif model_name=='ARIMA':
        Da=st.session_state['final_table']
        # print(Da['Actual'])
        act=Da['yact']
        pred=Da['Arima_pred']
        # st.line_chart(data=Da,x='month',y=['yact','Random_forest_pred'])
        plt.figure(figsize=(10, 5))
        plt.title('Actual VS Predicted')  
        plt.plot(Da['month'],Da['yact'],label='actual')
        plt.plot(Da['month'],Da['Arima_pred'],label='predicted')
        plt.xlabel('Month')
        plt.ylabel('Number Of Units')
        plt.legend()
        x_ticks = np.round(Da['month']).astype(int)
        plt.xticks(x_ticks)
        st.pyplot(plt)
    
    if outlet_code==None:
        Da1=pd.read_csv('my_file.csv')
        Da1=Da1[Da1['month']<=Date.month+duration]
        print(Da1)
        Da1=Da1.groupby('outlet_code').agg({'yact':'sum'})
        Da1['rank']=Da1.rank(method='first',ascending=False)
        Da1=Da1[Da1['rank']<=5]
        Da1.rename(columns={'yact':'no_of_sales'},inplace=True)
        Da1.sort_values('no_of_sales',ascending=False,inplace=True)
        Da1=Da1.reset_index()
        #  st.dataframe(Data)
        plt.figure(figsize=(8,7))
        plt.title('Top 5 Performing Outlets')
        plt.bar(Da1['outlet_code'].astype(str),Da1['no_of_sales'])
        Dat=Da1['no_of_sales'].astype(int)
        for i, value in enumerate(Dat):
            plt.text(i, value, str(value), ha='center', va='bottom')
        plt.xlabel('Outlet Code')
        plt.ylabel('Number Of Units Sold')
        st.pyplot(plt)
        # st.bar_chart(Da1,x='outlet_code',y='no_of_sales')
        # st.markdown("</div>", unsafe_allow_html=True)
# if st.session_state['Home']:
#    st.markdown(
#     """
#     <style>
#         .imagecon {
#         background-image:url('data:image/png;base64,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');
#         background-size:cover;
#             height: 50vh;
#             width: 80%;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }
#         .img1 {
#             max-width: 100%;
#             max-height: 100%;
#         }
#         .logo{
#         display:flex;
#         flex-direction:row;
#         justify-content:right;
#         align-items:center;
#         }
#     </style>
#     <div class='imagecon'>
#             <img class='img1' height='300' src=''>
#     </div>
#     <div class='logo' >
#     <img height='20' src='data:image/jpeg;base64,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'>
#     </div>
#    """,unsafe_allow_html=True)
   