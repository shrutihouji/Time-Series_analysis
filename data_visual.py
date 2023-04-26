import pandas as pd
import plotly.figure_factory as ff 
import plotly
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import mpld3
from numpy import array
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.subplots as sp


#Display table
def display_data():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)','Date']].dropna().head()
    df = ff.create_table(df1,height_constant=20)
    return plotly.offline.plot(df,output_type='div')


#display table
"""def display_data1():
    elec = pd.read_excel('Electricity.xlsx', sheet_name="Monthly Data", skiprows=10)
    elec['Month']=pd.to_datetime(elec['Month'])
    elec.index= pd.DatetimeIndex(elec['Month'])
    elec['month']=[elec.index[i].month for i in range(len(elec))]
    elec['year']=[elec.index[i].year for i in range(len(elec))]
    df1=elec[['Electricity Net Imports','Month']].dropna().head()
    df = ff.create_table(df1, height_constant=20)
    return plotly.offline.plot(df,output_type='div')"""


#display table
def display_data2():
    pet = pd.read_excel('Petroleum.xls', sheet_name="Data 1", skiprows=2)
    pet['Date']=pd.to_datetime(pet['Date'])
    pet.index= pd.DatetimeIndex(pet['Date'])
    pet['month']=[pet.index[i].month for i in range(len(pet))]
    pet['year']=[pet.index[i].year for i in range(len(pet))]
    df1=pet[['Weekly U.S. Field Production of Crude Oil  (Thousand Barrels per Day)', 'Date']].dropna().head()
    df = ff.create_table(df1, height_constant=20)
    return plotly.offline.plot(df,output_type='div')


def input():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', 'Date']].copy().dropna()
    return df1

def plot1():
    df1=input()
    fig = px.line(df1, x='Date', y='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)')
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
    title='Time Series with Rangeslider for Wellhead Price - Univariate TS PLOT',
    xaxis_title='Date',
    yaxis_title='Price')
    return plotly.offline.plot(fig, output_type='div')

def input_gas():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    return gas

def uni_plot2():
    gas=input_gas()
    fig = px.line(gas, x='month', y='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', color='year',
              title='Seasonal plot')
    fig.update_layout(legend=dict(title='Year', yanchor='top', y=0.99, xanchor='left', x=0.01))
    return plotly.offline.plot(fig,output_type='div')



def uni_plot3():
    df=input_gas()
    df=df[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    res1 = seasonal_decompose(df, model='multiplicative', period=80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res1.observed.index, y=res1.observed, name='Observed'))
    fig.add_trace(go.Scatter(x=res1.trend.index, y=res1.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=res1.seasonal.index, y=res1.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=res1.resid.index, y=res1.resid, name='Residual'))
    fig.update_layout(
            title="Multiplicative Decomposition",
            xaxis_title="Date",
            yaxis_title="Wellhead Price Value"
        )
    return plotly.offline.plot(fig,output_type='div')

def input3():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    return gas


"""def plot2():
    #seasonal plot using lineplot function
    gas=input2()
    fig=sns.lineplot(data=gas, 
             x='month', 
             y='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)', 
             hue='year', 
             legend='full')

    # add title
    plt.title('Seasonal plot')

    # move the legend outside of the main figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plot_html = mpld3.fig_to_html(plt.gcf())
    return plot_html"""

def input2():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Imports (MMcf)', 'Date', 'U.S. Natural Gas Exports (MMcf)']].copy().dropna()
    return df1

def plot2():
    df1=input2()
    fig = px.scatter(df1, x="U.S. Natural Gas Imports (MMcf)", y="U.S. Natural Gas Exports (MMcf)", 
                   color_continuous_scale=px.colors.sequential.Agsunset, render_mode="webgl",title="Scatter Plot Export Vs Imports for Natural Gas - Multivariate TS PLOT")
    return plotly.offline.plot(fig,output_type='div')

   
#Forecasting plots- ARIMA

def forecast_input():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    return df1


df1=forecast_input()
df1=df1.diff()
model = ARIMA(df1, order=(4,3,1))
model_fit = model.fit()
pickle.dump(model_fit, open("model2.pkl", "wb"))    


#Forecasting plots- VAR

def forecast_input2():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])

    df2=gas[['U.S. Natural Gas Imports (MMcf)','U.S. Natural Gas Exports (MMcf)']].copy()
    df2=df2.dropna()

    df_diff = df2.diff().dropna()
        
    import_export_df = df_diff[['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)']]

    return import_export_df

df_var=forecast_input2()
model_var = VAR(df_var[:-12])
model_fit_var = model_var.fit(maxlags = 13)
pickle.dump(model_fit_var,open("model3.pkl","wb"))


#NATURAL GAS- PRODUCTION

def prod_input():
    data = pd.read_excel('Price_NaturalGas.xls', sheet_name = "Data 2", skiprows = 2)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.index = pd.DatetimeIndex(data['Date'])
    df = data[['U.S. Natural Gas Gross Withdrawals (MMcf)']].copy()
    return df

#Lag Pot
def prod_plot1():
    df=prod_input()
    df_shift = pd.concat([df, df.shift()], axis=1)
    df_shift.columns = ["y", "y_lag1"]
    df_shift = df_shift.dropna()
    fig = px.scatter(df_shift, x="y_lag1", y="y", trendline="ols")
    fig.update_layout(
    title={
        'text': "Lag Plot of Natural Gas Production",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
    )
    return plotly.offline.plot(fig,output_type='div')

#additive decomposition plot
def prod_plot2():
    df=prod_input()
    res1 = seasonal_decompose(df, model='additive', period=40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res1.observed.index, y=res1.observed, name='Observed'))
    fig.add_trace(go.Scatter(x=res1.trend.index, y=res1.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=res1.seasonal.index, y=res1.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=res1.resid.index, y=res1.resid, name='Residual'))
    fig.update_layout(
        title="Additive Decomposition",
        xaxis_title="Date",
        yaxis_title="Production Value"
    )
    return plotly.offline.plot(fig,output_type='div')


#Consumption Visualizations

def con_input():
    data = pd.read_excel('Price_NaturalGas.xls', sheet_name = "Data 5", skiprows = 2)
    #drop na values 
    data = data.dropna()
    data.index = pd.DatetimeIndex(data['Date'])
    #Convert the Date column into a date object
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.drop(['Date', 'U.S. Natural Gas Total Consumption (MMcf)'] , axis = 1 )
    return df

def con_plot1():
    df=con_input()
    fig = px.line(df, x=df.index, y=df.columns, title='Consumption of Natural Gas over Different Sources')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Values')
    return plotly.offline.plot(fig,output_type='div')

def con_plot2():
    df=con_input()
    fig = px.box(data_frame=df, x=df.index.year, y=df.columns,
             title='Natural Gas Consumption by Year', color=df.index.year)
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Consumption of Natural Gas over different sources')
    return plotly.offline.plot(fig,output_type='div')


#ARIMA MODEL- Production

df_prod_arima=prod_input()
df_prod_arima=df_prod_arima.diff().dropna()
model = ARIMA(df_prod_arima, order=(4,1,3))
model_fit = model.fit()
pickle.dump(model_fit, open("model5.pkl", "wb"))


#VAR MODEL - Consumption
def forecast_input_var_con():
    data = pd.read_excel('Price_NaturalGas.xls', sheet_name = "Data 5", skiprows = 2)
    #drop na values 
    data = data.dropna()
    data.index = pd.DatetimeIndex(data['Date'])
    #Convert the Date column into a date object
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.drop(['Date', 'U.S. Natural Gas Total Consumption (MMcf)'] , axis = 1 )
    df_diff = df.diff().dropna()
    return df_diff
    

df_var_con=forecast_input_var_con()
training_set = df_var_con[:int(0.90*(len(df_var_con)))]
test_set = df_var_con[int(0.90*(len(df_var_con))):]
    
#Fit to a VAR model
model_var = VAR(endog=training_set)
#lags = model.select_order(maxlags=2)['aic']
model_fit_var_con = model_var.fit()
pickle.dump(model_fit_var_con,open("model5.pkl","wb"))


#LSTM
"""def input_forecast1():
    df=input()
    diff = df.diff().dropna()
    naturalgas_price_import = diff['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']
    values = naturalgas_price_import.values
    training_data_len = math.ceil(len(values)* 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(16, len(train_data)):
        x_train.append(train_data[i-16:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    test_data = scaled_data[training_data_len-16: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(16, len(test_data)):
        x_test.append(test_data[i-16:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()


    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size= 1, epochs=15)


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)


    data1 = diff.filter(['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'])
    train = data1[:training_data_len]
    validation = data1[training_data_len:]
    validation['Predictions'] = predictions

    fig = px.line()
    fig.add_scatter(x=train.index, y=train['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'], name='Train')
    fig.add_scatter(x=validation.index, y=validation['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)'], name='Val')
    fig.add_scatter(x=validation.index, y=validation['Predictions'], name='Predictions')
    fig.update_layout(title='Model', xaxis_title='Date', yaxis_title='U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)')
    return plotly.offline.plot(fig,output_type='div') """



        


