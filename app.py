from flask import Flask, render_template
from data_visual import display_data, display_data2, input, plot1,plot2,prod_plot1, prod_plot2, uni_plot2, uni_plot3, con_plot1, con_plot2
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import plotly.io as pio
#from sklearn.metrics import mean_squared_error 
from math import sqrt


app=Flask(__name__)

@app.route("/Dashboard")
def home():
    return render_template("main.html")

@app.route('/Dashboard/Resources')
def hello():
    data=display_data()
    #data1=display_data1()
    data2=display_data2()
    return render_template('index.html', returnList = data, returnList2=data2)

@app.route('/Dashboard/Resources/Visualizations')
def visual():
    visual1=plot1()
    visual2=uni_plot3()
    return render_template('visuals.html', visual_1=visual1, visual_2=visual2)


@app.route('/Dashboard/Resources/Visualizations1')
def visual2():
    visual1=uni_plot2()
    visual2=plot2()
    return render_template('visuals1.html', visual_1=visual1, visual_2=visual2)

#ARIMA MODEL
#Load the pickle file
model=pickle.load(open("model2.pkl","rb"))
def input():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 1", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])
    gas['month']=[gas.index[i].month for i in range(len(gas))]
    gas['year']=[gas.index[i].year for i in range(len(gas))]
    df1=gas[['U.S. Natural Gas Wellhead Price (Dollars per Thousand Cubic Feet)']].copy().dropna()
    return df1 


@app.route('/Dashboard/Resources/Forecasting')
def forecast_plot():
    df1=input()
    df1=df1.diff()
    X = df1.values
    size = int(len(X) * 0.80)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(4,3,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    fig = px.line(title='ARIMA Model Performance')
    fig.add_scatter(x=list(range(len(test))), y=test.flatten(), mode='lines', name='Actual')
    fig.add_scatter(x=list(range(len(predictions))), y=predictions, mode='lines', name='Predicted')
    plot_html = pio.to_html(fig, full_html=False)
    n_steps = 60 # for example, to predict next 5 months
    forecast = model_fit.forecast(steps=n_steps)

    # Plot original data with forecast
    fig1 = px.line(df1, title='ARIMA Forecast')
    fig1.add_scatter(x=pd.date_range(start=df1.index[-1], periods=n_steps+1, freq='M')[1:], y=forecast, mode='lines', name='Forecast')
    plot_html1 = pio.to_html(fig1, full_html=False)
    return render_template('forecast.html', forecast1=plot_html, forecast2=plot_html1)

 
#VAR MODEL
model3=pickle.load(open("model3.pkl","rb"))
def input1():
    gas = pd.read_excel('Price_NaturalGas.xls', sheet_name="Data 3", skiprows=2)
    gas['Date']=pd.to_datetime(gas['Date'])
    gas.index= pd.DatetimeIndex(gas['Date'])

    df2=gas[['U.S. Natural Gas Imports (MMcf)','U.S. Natural Gas Exports (MMcf)']].copy()
    df2=df2.dropna()

    df_diff = df2.diff().dropna()

    training_set = df_diff[:int(0.95*(len(df2)))]
    test_set = df_diff[int(0.95*(len(df2))):]
        
    import_export_df = df_diff[['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)']]
    return test_set,import_export_df,df_diff


@app.route('/Dashboard/Resources/Forecasting1')
def forecast_plot1():
    test_set,import_export_df,df_diff=input1()
    pred = model3.forecast(test_set.values, steps=len(test_set))
    aic_var = model3.aic
    bic_var = model3.bic

    forecast = pd.DataFrame(pred, index=test_set.index, columns=test_set.columns)
    #rmse = sqrt(mean_squared_error(test_set, forecast))


    # Create traces for actual and predicted values
    actual_trace = go.Scatter(x=test_set.index, y=test_set['U.S. Natural Gas Imports (MMcf)'],
                                mode='lines', name='Actual')
    predicted_trace = go.Scatter(x=forecast.index, y=forecast['U.S. Natural Gas Imports (MMcf)'],
                                mode='lines', name='Predicted')

    # Create the figure and add traces
    fig1 = go.Figure()
    fig1.add_trace(actual_trace)
    fig1.add_trace(predicted_trace)

    # Add title and axis labels
    fig1.update_layout(title='Actual vs Predicted U.S. Natural Gas Imports (MMcf)',
                        xaxis_title='Date', yaxis_title='U.S. Natural Gas Imports (MMcf)')

  
    plot_html = fig1.to_html(full_html=False)

 
    forecast_df = pd.DataFrame(forecast, index=test_set.index, columns=test_set.columns)

  # Invert the differenced values to get the actual gas imports and exports data
    forecast_df['U.S. Natural Gas Imports (MMcf)'] = forecast_df['U.S. Natural Gas Imports (MMcf)'].cumsum() + import_export_df['U.S. Natural Gas Imports (MMcf)'].iloc[-121]
    forecast_df['U.S. Natural Gas Exports (MMcf)'] = forecast_df['U.S. Natural Gas Exports (MMcf)'].cumsum() + import_export_df['U.S. Natural Gas Exports (MMcf)'].iloc[-121]


    # Plot the forecasted values
    import plotly.express as px

    fig2 = px.line(df_diff, y=['U.S. Natural Gas Imports (MMcf)', 'U.S. Natural Gas Exports (MMcf)'], 
                    title='Historical Imports and Exports')

    fig2.add_scatter(x=forecast_df.index, y=forecast_df['U.S. Natural Gas Imports (MMcf)'], mode='lines', 
                    name='Forecasted Imports')

    fig2.add_scatter(x=forecast_df.index, y=forecast_df['U.S. Natural Gas Exports (MMcf)'], mode='lines', 
                    name='Forecasted Exports')

    plot_html2 = fig2.to_html(full_html=False)

    return render_template('forecast_new.html', forecast_new=plot_html, forecast_new2=plot_html2)
    # Display the plot
    #fig1.show()

    #plot_html2 = pio.to_html(fig1, full_html=False)
    #return render_template('forecast.html', forecast2=plot_html2)

    # Display the plot


#NATURAL GAS - PRODUCTION
@app.route('/Dashboard/Resources/Visualizations_Production')
def visual_production():
    visual1=prod_plot1()
    visual2=prod_plot2()
    return render_template('visuals_production.html', visual_1=visual1, visual_2=visual2)

#natural Gas-Consumption
@app.route('/Dashboard/Resources/Visualizations_Consumption')
def visual_consumption():
    visual1=con_plot1()
    visual2=con_plot2()
    return render_template('visuals_consumption.html', visual_1=visual1, visual_2=visual2)



#ARIMA MODEL - Production
#Load the pickle file
model4=pickle.load(open("model4.pkl","rb"))

def input_prod():
    data = pd.read_excel('Price_NaturalGas.xls', sheet_name = "Data 2", skiprows = 2)
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data.index = pd.DatetimeIndex(data['Date'])
    df = data[['U.S. Natural Gas Gross Withdrawals (MMcf)']].copy()
    return df 

@app.route('/Dashboard/Resources/Forecasting_Production')
def forecast_plot_prod():
    df1=input_prod()
    df1=df1.diff()
    X = df1.values
    size = int(len(X) * 0.80)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()


    # walk-forward validation
    for t in range(len(test)):
        model4 = ARIMA(history, order=(4,1,3))
        model_fit = model4.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    fig = px.line(title='ARIMA Model Performance')
    fig.add_scatter(x=list(range(len(test))), y=test.flatten(), mode='lines', name='Actual')
    fig.add_scatter(x=list(range(len(predictions))), y=predictions, mode='lines', name='Predicted')

    plot_html = pio.to_html(fig, full_html=False)


    n_steps = 60 # for example, to predict next 5 months
    forecast = model_fit.forecast(steps=n_steps)

    # Plot original data with forecast
    fig1 = px.line(df1, title='ARIMA Forecast')
    fig1.add_scatter(x=pd.date_range(start=df1.index[-1], periods=n_steps+1, freq='M')[1:], y=forecast, mode='lines', name='Forecast')
    plot_html1 = pio.to_html(fig1, full_html=False)
    return render_template('forecast_production.html', forecast1=plot_html, forecast2=plot_html1)



#VAR MODEL-Consumption

model5=pickle.load(open("model5.pkl","rb"))
def forecast_input_var_con():
    data = pd.read_excel('Price_NaturalGas.xls', sheet_name = "Data 5", skiprows = 2)
    #drop na values 
    data = data.dropna()
    data.index = pd.DatetimeIndex(data['Date'])
    #Convert the Date column into a date object
    data['Date'] = pd.to_datetime(data['Date'])
    df = data.drop(['Date', 'U.S. Natural Gas Total Consumption (MMcf)'] , axis = 1 )
    df_diff = df.diff().dropna()

    training_set = df_diff[:int(0.90*(len(df_diff)))]
    test_set = df_diff[int(0.90*(len(df_diff))):]
    return test_set, df_diff, data,df


@app.route('/Dashboard/Resources/Forecasting_Consumption')
def forecast_plot_con():
    test_set, df_diff, data, df=forecast_input_var_con()
    pred = model5.forecast(test_set.values, steps = 27)
    forecast = pd.DataFrame(pred, index=test_set.index, columns=test_set.columns)
    import plotly.express as px
    fig = px.line()
    fig.add_scatter(x=test_set.index, y=test_set.iloc[:,1], name='actual')
    fig.add_scatter(x=test_set.index, y=forecast.iloc[:,1], name='forecast')
    plot_html= pio.to_html(fig, full_html=False)

    n_steps = 36
    pred1 = model5.forecast(df_diff.values[-model5.k_ar:], steps=n_steps)
    forecast = pd.DataFrame(pred1, index=pd.date_range(start=data.index[-1], periods=n_steps+1, freq='MS')[1:], columns=df.columns)

    # Combine the actual and forecasted data into a single DataFrame
    combined_df = pd.concat([df_diff, forecast], axis=0)

    # Plot the actual and forecasted data using Plotly Express
    fig1 = px.line(combined_df, x=combined_df.index, y=combined_df.columns, labels={'value': 'Compumption'})
    fig1.update_layout(title='Natural Gas Consumption Forecast for the next 3 years', xaxis_title='Date')
    fig1.update_traces(mode='lines')
    plot_html1= pio.to_html(fig1, full_html=False)
    return render_template('forecast_consumption.html', forecast1=plot_html, forecast2=plot_html1)

