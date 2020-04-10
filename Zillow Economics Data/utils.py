import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
import seaborn as sns
sns.set(style = 'whitegrid')
init_notebook_mode()

def missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False) # getting the sum of null values and ordering
    nullYear=data.isnull().groupby([data.year]).sum() # sum of null values grouped by year
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) #getting the percent and order of null
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Concatenating the total and percent
    print("Total number of columns: " , len(data.columns))
    print("Columns with null values: ", len(df[~(df['Total'] == 0)]))
    print (df[~(df['Total'] == 0)]) # Returning values of nulls different of 0
    return df,nullYear


def analysisChart(dataset,column, Title):
    uniqueStates = set(dataset[
                           ~dataset[column].isnull()]['RegionName'].values)

    dfStates = dataset[dataset['RegionName'].isin(uniqueStates)].copy()
    highestStates = dfStates[['RegionName', column]].groupby('RegionName').max().sort_values(by=[column],
                                                                                             ascending=False)[
                    :6].index.values.tolist()
    dfStates = dfStates[dfStates.RegionName.isin(highestStates)]
    dfStates.year = dfStates.Date.dt.year
    meanState = dfStates.groupby([dfStates.year, dfStates.RegionName])[column].mean().dropna().reset_index(name=column)
    meanState = meanState.pivot(index='year', columns='RegionName', values=column)

    labels = meanState.columns
    colors = ['rgb(67,67,67)', 'rgb(235, 73, 52)', 'rgb(252, 186, 3)', 'rgb(49,130,189)', 'rgb(189,189,189)']
    fig = go.Figure()
    annotations = []
    for i in range(0, 5):
        fig.add_trace(go.Scatter(x=meanState.index, y=meanState[labels[i]], mode='lines',
                                 line=dict(color=colors[i]), name=labels[i]
                                 ))

    fig.layout.update(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),

        autosize=True,
        margin=dict(
            autoexpand=True,
            l=20,
            r=20,
            t=110,
        ),
        showlegend=True,
        plot_bgcolor='white'
    )

    # Source
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                            xanchor='center', yanchor='top',
                            text='Data Source: Zillow',
                            font=dict(family='Arial',
                                      size=12,
                                      color='rgb(150,150,150)'),
                            showarrow=False))

    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                            xanchor='left', yanchor='bottom',
                            text=Title,
                            font=dict(family='Arial',
                                      size=30,
                                      color='rgb(37,37,37)'),
                            showarrow=False))
    fig.layout.update(annotations=annotations)
    fig.show()

def columns_in(dataframe_columns,name):
    return [item for item in dataframe_columns if name in item]

def corrChart(dataframe,columns):
    corrmat = dataframe[columns].corr()
    corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(dataframe[corr_features].corr(),annot=True,cmap="RdYlGn")


def fill_missing_values(data,column):
    data['month_year'] = pd.to_datetime(data.index).to_period('M')
    data[column]=data.groupby(['monthYear'],sort=False)[column].apply(lambda x: x.fillna(x.median()))
    return data[column]

