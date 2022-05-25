import pandas as pd
import numpy as np

# preprocess the covid data from https://www.bts.gov/daily-travel
# and travel data from https://github.com/nytimes/covid-19-data/tree/master/rolling-averages.


#--------------------------------------------#
#----------------covid process---------------#
#--------------------------------------------#
data20 = pd.read_csv('us-counties-2020.csv')
data21 = pd.read_csv('us-counties-2021.csv')
data = pd.concat([data20, data21])

data.dropna(inplace=True) 
# We retrain the 7 days average smooth version of the newly confirmed cases and deaths 
data.drop(['cases', 'deaths', 'cases_avg_per_100k', 'deaths_avg_per_100k'], axis=1,inplace=True)

data.rename({'cases_avg':'cases', 'deaths_avg':'deaths'}, axis=1, inplace=True)
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')
StateCounty = data.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
data.insert(0, 'StateCounty', StateCounty)
fips = data.apply(lambda x:x['geoid'][4:], axis=1)
data.insert(2, 'fips', fips)
data.drop(['geoid'], axis=1, inplace=True)
data.sort_values(by = ['date'], axis = 0,inplace=True)
data.reset_index(drop=True, inplace=True)

# drop data which is NaN in travel data
data.drop(data[data.fips.isin([15005,48301])].index, axis=0, inplace=True)
data.sort_values(by=['date'], axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)
data.to_csv('./data/us-counties-smooth.csv', index=False)


#--------------------------------------------#
#----------------travel process---------------#
#--------------------------------------------#
travelDistanceDf = pd.read_csv('./data/Trips_by_Distance.csv')
colName = travelDistanceDf.columns.tolist()

travelDistanceDf = travelDistanceDf[travelDistanceDf['Level'] == 'County']
travelDistanceDf.rename({'Date':'date'}, axis=1, inplace=True)
travelDistanceDf['date'] = pd.to_datetime(travelDistanceDf['date'], format = '%Y-%m-%d')
travelDistanceDf.drop(['Level', 'State FIPS', 'State Postal Code'], axis=1, inplace=True)
travelDistanceDf.sort_values(by=['date'], axis=0, inplace=True)
tempTime = travelDistanceDf['date'].unique()


dataDF = pd.read_csv('./data/us-counties-smooth.csv')
dataDF.loc[dataDF['county'] == 'New York City', 'fips'] = 36061 
dataDF['county'] = dataDF['county'].map(lambda x: x[:-4]+'City' if x[-4:] == 'city' else x)
dataDF.loc[dataDF['county'] == 'New York City', 'county'] = 'New York'
dataDF.loc[dataDF['county'] == 'Fairbanks North Star Borough', 'county'] = 'Fairbanks North Star'
dataDF.loc[dataDF['county'] == 'Ketchikan Gateway Borough', 'county'] = 'Ketchikan Gateway'
dataDF.loc[dataDF['county'] == 'Kenai Peninsula Borough', 'county'] = 'Kenai Peninsula'
dataDF.loc[dataDF['county'] == 'Doña Ana', 'county'] = 'Dona Ana'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Juneau City and Borough'), 'county'] = 'Juneau'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Matanuska-Susitna Borough'), 'county'] = 'Matanuska-Susitna'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Bethel Census Area'), 'county'] = 'Bethel'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Kodiak Island Borough'), 'county'] = 'Kodiak Island'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Nome Census Area'), 'county'] = 'Nome'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Petersburg Borough'), 'county'] = 'Petersburg'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Prince of Wales-Hyder Census Area'), 'county'] = 'Prince of Wales-Hyder'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Southeast Fairbanks Census Area'), 'county'] = 'Southeast Fairbanks'
dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Yukon-Koyukuk Census Area'), 'county'] = 'Yukon-Koyukuk'
dataDF.rename(columns = {'fips': 'County FIPS'}, inplace = True)

stateDF = dataDF['County FIPS'].value_counts()
stateDF = pd.DataFrame(stateDF).reset_index()
stateDF.rename(columns = {'index':'County FIPS', 'County FIPS':'count'}, inplace = True)
stateDF = stateDF[stateDF['County FIPS'].isin(travelDistanceDf['County FIPS'])]
stateDF.reset_index(drop = True, inplace = True)

# time aligned
dataDF['date'] = pd.to_datetime(dataDF['date'], format = '%Y-%m-%d')
minTime = pd.to_datetime(min(dataDF.date.unique()), format = '%Y-%m-%d')
maxTime = pd.to_datetime(max(travelDistanceDf.date.unique()), format = '%Y-%m-%d')
dataDF = dataDF[dataDF['date'] <= maxTime]
travel = travelDistanceDf[travelDistanceDf['date'] >= minTime]

travel.drop(['Row ID', 'Week'], axis=1, inplace=True)
travel.insert(1, 'month', travel['Month'])
travel.sort_values(by=['County Name', 'date'], axis=0, inplace=True)
travel.reset_index(drop=True, inplace=True)
travel.drop(travel[travel['County FIPS']==15005.0].index, axis=0, inplace=True)

renameCol = [
     'Population Staying at Home',
     'Population Not Staying at Home',
     'Number of Trips',
     'Number of Trips <1',
     'Number of Trips 1-3',
     'Number of Trips 3-5',
     'Number of Trips 5-10',
     'Number of Trips 10-25',
     'Number of Trips 25-50',
     'Number of Trips 50-100',
     'Number of Trips 100-250',
     'Number of Trips 250-500',
     'Number of Trips >=500',
]

# use the mean of the travel data to fill the NaN for each county separately
df = travel.groupby(['County FIPS']).mean()
df.loc[:,renameCol] = df.loc[:,renameCol].astype(np.int32)
nan_ind = np.where(travel['Number of Trips'].isnull())[0]
x = df.copy()
for ind in nan_ind:
    print(ind)
    fip = travel.iloc[ind]['County FIPS']
    for col in renameCol:
        travel.loc[ind, col] = x.loc[fip,col]
    
countyFips = dataDF[['StateCounty', 'County FIPS']]
countyFips.drop_duplicates(inplace = True)
travelDistanceDf = pd.merge(travel, countyFips, on = ['County FIPS'], how = 'left')

travelDistanceDf = travelDistanceDf[~travelDistanceDf['StateCounty'].isna()]
StateCounty = travelDistanceDf['StateCounty']
travelDistanceDf.drop(columns = ['StateCounty'], inplace = True)
travelDistanceDf.insert(0, 'StateCounty', StateCounty)
del StateCounty
del dataDF

travelDistanceDf.reset_index(drop=True, inplace=True)
travelDistanceDf.to_csv('./data/travel_processed.csv',index=False)

# merge covid and travel data
travel.drop(columns = ['County Name'], inplace = True)
travel.rename(columns = {'County FIPS': 'fips'}, inplace = True)
travel['date'] = pd.to_datetime(travel['date'], format = '%Y-%m-%d')
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')
colname_travel = travel.columns.tolist()
dataset = pd.merge(data, travel, on = ['StateCounty', 'date', 'fips'], how = 'inner')
dataset.reset_index(drop = True, inplace = True)
fips = dataset.fips
dataset.drop(columns = ['fips'], inplace = True)
dataset.insert(1, 'fips', fips)
dataset.to_csv('./data/counties-covid-travel.csv', index = False)