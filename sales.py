import pandas as pd
import plotly.graph_objs as go
sales=pd.read_csv("284_618_bundle_archive/vgsales.csv")
sales.info()
from plotly.offline import iplot
df=sales.head(100)
#按地区观察销量可视化
trace1 = go.Scatter(
                    x = df.Rank,
                    y = df.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),
                    text= df.Name)

trace2 = go.Scatter(
                    x = df.Rank,
                    y = df.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),
                    text= df.Name)
trace3 = go.Scatter(
                    x = df.Rank,
                    y = df.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',size=8),
                    text= df.Name)
trace4 = go.Scatter(
                    x = df.Rank,
                    y = df.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'lime',size=8),
                    text= df.Name)
                    

data = [trace1, trace2,trace3,trace4]
layout = dict(title = 'North America, Europe, Japan and Other Sales of Top 100 Video Games',
              xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'Sales(In Millions)',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)
#根据全球销售量及其发行商来查看前一百名游戏的发行年份可视化
fig={
    "data" : [
    {
        'x': df.Rank,
        'y': df.Year,
        'mode': 'markers',
        'marker': {
            "color":df.Global_Sales,
            'size': df.Global_Sales,
            'showscale': True,
            "colorscale":'Blackbody'
        },
        "text" :  "Name:"+ df.Name +","+" Publisher:" + df.Publisher
        
    },
],
"layout":
    {
    "title":"Release Years of Top 100 Video Games According to Global Sales",
    "xaxis":{
        "title":"Rank",
        "gridcolor":'rgb(255, 255, 255)',
        "zerolinewidth":1,
        "ticklen":5,
        "gridwidth":2,
    },
    "yaxis":{
        "title":'Years',
        "gridcolor":'rgb(255, 255, 255)',
        "zerolinewidth":1,
        "ticklen":5,
        "gridwidth":2,
    },
    
    "paper_bgcolor":'rgb(243, 243, 243)',
    "plot_bgcolor":'rgb(243, 243, 243)'
    }}

iplot(fig)
#前100个游戏的发行商数量可视化
trace = go.Histogram(x=df.Publisher,marker=dict(color="crimson",line=dict(color='black', width=2)),opacity=0.75)
layout = go.Layout(
    title='Numbers of Top 100 Video Games Publishers',
    xaxis=dict(
        title='Publishers'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
import numpy as np
xaction=sales[sales.Genre=="Action"]
xsports=sales[sales.Genre=="Sports"]
xmisc=sales[sales.Genre=="Misc"]
xrole=sales[sales.Genre=="Role-Playing"]
xshooter=sales[sales.Genre=="Shooter"]
xadventure=sales[sales.Genre=="Adventure"]
xrace=sales[sales.Genre=="Racing"]
xplatform=sales[sales.Genre=="Platform"]
xsimulation=sales[sales.Genre=="Simulation"]
xfight=sales[sales.Genre=="Fighting"]
xstrategy=sales[sales.Genre=="Strategy"]
xpuzzle=sales[sales.Genre=="Puzzle"]
#类型和平台的角度来看游戏的销售可视化
trace1 = go.Histogram(
    x=xaction.Platform,
    opacity=0.75,
    name = "Action",
    marker=dict(color='rgb(165,0,38)'))
trace2 = go.Histogram(
    x=xsports.Platform,
    opacity=0.75,
    name = "Sports",
    marker=dict(color='rgb(215,48,39)'))
trace3 = go.Histogram(
    x=xmisc.Platform,
    opacity=0.75,
    name = "Misc",
    marker=dict(color='rgb(244,109,67)'))
trace4 = go.Histogram(
    x=xrole.Platform,
    opacity=0.75,
    name = "Role Playing",
    marker=dict(color='rgb(253,174,97)'))
trace5 = go.Histogram(
    x=xshooter.Platform,
    opacity=0.75,
    name = "Shooter",
    marker=dict(color='rgb(254,224,144)'))
trace6 = go.Histogram(
    x=xadventure.Platform,
    opacity=0.75,
    name = "Adventure",
    marker=dict(color='rgb(170,253,87)'))
trace7 = go.Histogram(
    x=xrace.Platform,
    opacity=0.75,
    name = "Racing",
    marker=dict(color='rgb(171,217,233)'))
trace8 = go.Histogram(
    x=xplatform.Platform,
    opacity=0.75,
    name = "Platform",
    marker=dict(color='rgb(116,173,209)'))
trace9 = go.Histogram(
    x=xsimulation.Platform,
    opacity=0.75,
    name = "Simulation",
    marker=dict(color='rgb(69,117,180)'))
trace10 = go.Histogram(
    x=xfight.Platform,
    opacity=0.75,
    name = "Fighting",
    marker=dict(color='rgb(49,54,149)'))
trace11 = go.Histogram(
    x=xstrategy.Platform,
    opacity=0.75,
    name = "Strategy",
    marker=dict(color="rgb(10,77,131)"))
trace12 = go.Histogram(
    x=xpuzzle.Platform,
    opacity=0.75,
    name = "Puzzle",
    marker=dict(color='rgb(1,15,139)'))

data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]
layout = go.Layout(barmode='stack',
                   title='Genre Counts According to Platform',
                   xaxis=dict(title='Platform'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Bar(
    x=xaction.groupby("Platform")["Global_Sales"].sum().index,
    y=xaction.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Action",
    marker=dict(color="rgb(119,172,238)"))
trace2 = go.Bar(
    x=xsports.groupby("Platform")["Global_Sales"].sum().index,
    y=xsports.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Sports",
    marker=dict(color='rgb(21,90,174)'))
trace3 = go.Bar(
    x=xrace.groupby("Platform")["Global_Sales"].sum().index,
    y=xrace.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Racing",
    marker=dict(color="rgb(156,245,163)"))
trace4 = go.Bar(
    x=xshooter.groupby("Platform")["Global_Sales"].sum().index,
    y=xshooter.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Shooter",
    marker=dict(color="rgb(14,135,23)"))
trace5 = go.Bar(
    x=xmisc.groupby("Platform")["Global_Sales"].sum().index,
    y=xmisc.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Misc",
    marker=dict(color='rgb(252,118,103)'))
trace6 = go.Bar(
    x=xrole.groupby("Platform")["Global_Sales"].sum().index,
    y=xrole.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Role Playing",
    marker=dict(color="rgb(226,28,5)"))
trace7 = go.Bar(
    x=xfight.groupby("Platform")["Global_Sales"].sum().index,
    y=xfight.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Fighting",
    marker=dict(color="rgb(247,173,13)"))
trace8 = go.Bar(
    x=xplatform.groupby("Platform")["Global_Sales"].sum().index,
    y=xplatform.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Platform",
    marker=dict(color="rgb(242,122,13)"))
trace9 = go.Bar(
    x=xsimulation.groupby("Platform")["Global_Sales"].sum().index,
    y=xsimulation.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Simulation",
    marker=dict(color="rgb(188,145,202)"))
trace10 = go.Bar(
    x=xadventure.groupby("Platform")["Global_Sales"].sum().index,
    y=xadventure.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Adventure",
    marker=dict(color='rgb(104,57,119)'))
trace11 = go.Bar(
    x=xstrategy.groupby("Platform")["Global_Sales"].sum().index,
    y=xstrategy.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Strategy",
    marker=dict(color='rgb(245,253,104)'))
trace12 = go.Bar(
    x=xpuzzle.groupby("Platform")["Global_Sales"].sum().index,
    y=xpuzzle.groupby("Platform")["Global_Sales"].sum().values,
    opacity=0.75,
    name = "Puzzle",
    marker=dict(color='rgb(138,72,40)'))

data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]
layout = go.Layout(barmode='stack',
                   title='Total Global Sales According to Platform and Genre',
                   xaxis=dict(title='Platform'),
                   yaxis=dict( title='Global Sales(In Millions)'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#预测全球销售量
df=pd.read_csv("284_618_bundle_archive/vgsales.csv")
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['Name']=label.fit_transform(df['Name'])
df['Platform']=label.fit_transform(df['Platform'])
df['Genre']=label.fit_transform(df['Genre'])
df['Publisher']=label.fit_transform(df['Publisher'])
df['Global_Sales'] = df['Global_Sales'].astype(int)
df =df.drop(['NA_Sales', 'EU_Sales', 'JP_Sales','Other_Sales'], axis=1)
from sklearn.model_selection import train_test_split
train, test=train_test_split(df, test_size=0.1, random_state=1)

def data_splitting(df):
    x=df.drop(['Global_Sales'], axis=1)
    y=df['Global_Sales']
    return x, y

x_train, y_train=data_splitting(train)
x_test, y_test=data_splitting(test)
from sklearn.linear_model import LinearRegression
log = LinearRegression()
log.fit(x_train , y_train)
log_train = log.score(x_train , y_train)
log_test = log.score(x_test , y_test)

print("Training score :" , log_train)
print("Testing score :" , log_test)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
smote = XGBClassifier()
smote.fit(x_train, y_train)

# Predict on test
smote_pred = smote.predict(x_test)
accuracy = accuracy_score(y_test, smote_pred)
print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from IPython.display import display
%matplotlib inline
data=pd.read_csv("284_618_bundle_archive/vgsales.csv")
data = data[np.isfinite(data['Year'])]
naSales = data['NA_Sales']
#划分特征
features = data.drop(['Name', 'Global_Sales', 'NA_Sales'], axis = 1)
salesFeatures = features.drop(['Rank', 'Platform', 'Year', 'Genre', 'Publisher'], 
                              axis = 1)
otherFeatures = features.drop(['EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'], 
                              axis = 1)
temp = pd.DataFrame(index = rebuiltFeatures.index)

for col, col_data in rebuiltFeatures.iteritems():
    
    if col_data.dtype == object:
        col_data = pd.get_dummies(col_data, prefix = col)
        
    temp = temp.join(col_data)
    
rebuiltFeatures = temp
display(rebuiltFeatures[:5])
# Dividing the data into training and testing sets...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rebuiltFeatures, 
                                                    naSales, 
                                                    test_size = 0.2, 
                                                    random_state = 2)
# Creating & fitting a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

regDTR = DecisionTreeRegressor(random_state = 4)
regDTR.fit(X_train, y_train)
y_regDTR = regDTR.predict(X_test)

from sklearn.metrics import r2_score
print ('The following is the r2_score on the DTR model...')
print (r2_score(y_test, y_regDTR))

# Creating a K Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor

regKNR = KNeighborsRegressor()
regKNR.fit(X_train, y_train)
y_regKNR = regKNR.predict(X_test)
#比较两个模型
print ('The following is the r2_score on the KNR model...')
print (r2_score(y_test, y_regKNR))

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
pca.fit(salesFeatures)
salesFeaturesTransformed = pca.transform(salesFeatures)

salesFeaturesTransformed = pd.DataFrame(data = salesFeaturesTransformed, 
                                        index = salesFeatures.index, 
                                        columns = ['Sales'])
rebuiltFeatures = pd.concat([otherFeatures, salesFeaturesTransformed], 
                            axis = 1)

display(rebuiltFeatures[:5])
