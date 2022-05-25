import pandas as pd
from datetime import timedelta
import numpy as np
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Generate the data for training.')
parser.add_argument('--enc_steps', default=30, type=int, help='days data to encode')
parser.add_argument('--pred_steps', default=7, type=int, help='days expected to predict')
parser.add_argument('--use_month', action='store_true', help='use the month information in the model')
parser.add_argument('--agg_travel', action='store_true', 
                    help='aggregate 10 variables about travel to 3(close, middle, distant)')


def main():
    args = parser.parse_args()
    data = pd.read_csv('./data/covid-travel-new.csv')
    data['month'] = data['month'] / 12
    fips = data['fips'].unique().tolist()
    if args.agg_travel:
        for c in ['close', 'middle', 'distant']:
            data[c] = None
        data['close'] = data.iloc[:, -13:-9].sum(axis=1)
        data['middle'] = data.iloc[:, -9:-6].sum(axis=1)
        data['distant'] = data.iloc[:, -6:-3].sum(axis=1)
        data.drop(columns=data.columns.tolist()[-16:-3], axis=1, inplace=True)
        travelName = data.columns.tolist()[-3:]
    else:
        data.drop(columns=data.columns.tolist()[-13:-10], axis=1, inplace=True)
        travelName = data.columns.tolist()[-10:]

    colNames = ['StateCounty','fips','date']+[f'month_{i+1}' for i in range(args.enc_steps)]\
        + [f'enc_cases{i+1}' for i in range(args.enc_steps)] \
        + [f'enc_deaths{i+1}' for i in range(args.enc_steps)] \
        + [f'{col}_{i+1}' for col in travelName for i in range(args.enc_steps)] \
        + [f'pred_month_{i+1}' for i in range(args.pred_steps)]\
        + [f'pred_cases{i+1}' for i in range(args.pred_steps)] \
        + [f'pred_deaths{i+1}' for i in range(args.pred_steps)]

    if not args.use_month:
        data.drop(columns=['month'], axis=1, inplace=True)
        colNames = ['StateCounty','fips','date'] \
            + [f'enc_cases{i+1}' for i in range(args.enc_steps)] \
            + [f'enc_deaths{i+1}' for i in range(args.enc_steps)] \
            + [f'{col}_{i+1}' for col in travelName for i in range(args.enc_steps)] \
            + [f'pred_cases{i+1}' for i in range(args.pred_steps)] \
            + [f'pred_deaths{i+1}' for i in range(args.pred_steps)]
    
    encoder_dim = 2 + 1 * args.use_month + 3 + 7 * (not args.agg_travel)
    sub_fips = data[data['state']=='California'].fips.unique().tolist()
    #sub_fips = data.fips.unique().tolist()

    # train data
    train_list = []
    for fip in sub_fips:
        df = data.loc[data['fips']==fip]
        date = df['date'].unique()
        date.sort()
        train_data = df[df['date'].isin(date[:-args.pred_steps])]
        train_data['date'] = pd.to_datetime(train_data['date'])
        train_date = train_data['date'].unique()
        train_date.sort()
        train_data.reset_index(inplace = True, drop = True)
        trainDF = train_data[train_data['date'].isin(train_date[:-(args.enc_steps+args.pred_steps)+1])]
        renameCol = ['month', 'cases', 'deaths'] + travelName if args.use_month else ['cases', 'deaths'] + travelName
        renameDict = {x: x+'_1' for x in renameCol}
        for x in renameCol:
            for i in range(1, args.enc_steps+1):
                trainDF['{}_{}'.format(x, i)] = None
        for x in renameCol:
            trainDF[renameDict[x]] = trainDF[x]
        predCol = ['month', 'cases', 'deaths'] if args.use_month else ['cases', 'deaths']
        for x in ['pred_'+ c for c in  predCol]:
            for i in range(1, args.pred_steps+1):
                trainDF['{}{}'.format(x, i)] = None
        trainDF.drop(renameCol+['state','county'], axis=1, inplace=True)

        for i in range(1, args.enc_steps):
            nextDF = train_data[train_data['date'].isin(train_date[i:-(args.enc_steps+args.pred_steps)+1+i])]
            nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
            trainDF = trainDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
            nextDF.reset_index(inplace = True, drop = True)
            trainDF.reset_index(inplace = True, drop = True)
            temp = nextDF[(nextDF['date'] == trainDF['date'] + timedelta(i)) & \
                    (nextDF['StateCounty'] == trainDF['StateCounty'])]
            for x in renameCol:
                trainDF[f'{x}_{i+1}'] = temp[x]

        for j in range(args.pred_steps):
            if -args.pred_steps+j+1 == 0:
                nextDF = train_data[train_data['date'].isin(train_date[args.enc_steps+j:])]
            else:
                nextDF = train_data[train_data['date'].isin(train_date[args.enc_steps+j:-args.pred_steps+j+1])]
            nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
            trainDF = trainDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
            nextDF.reset_index(inplace = True, drop = True)
            trainDF.reset_index(inplace = True, drop = True)
            temp = nextDF[(nextDF['date'] == trainDF['date'] + timedelta(args.enc_steps+j)) & \
                    (nextDF['StateCounty'] == trainDF['StateCounty'])]
            for x in predCol:
                trainDF[f'pred_{x}{j+1}'] = temp[x]
        trainDF.loc[:, 'date'] = trainDF.loc[:, 'date'] + timedelta(args.enc_steps) #pred_cases1
        train_list.append(trainDF)
    traindf = pd.concat(train_list)
    traindf.reset_index(drop=True, inplace=True)
    traindf.to_csv(f'./useless/Cal_train{encoder_dim}_{args.pred_steps}.csv', index=False)

    # test data (last 7 days data)
    testdf = pd.DataFrame(columns=colNames)
    for fip in sub_fips:
        df = data.loc[data['fips']==fip]
        date = df['date'].unique()
        date.sort()
        test_data = df[df['date'].isin(date[-(args.pred_steps+args.enc_steps):])]
        statecounty = test_data['StateCounty'].unique().tolist()
        test_enc = test_data.iloc[:args.enc_steps]
        test_target = test_data.iloc[-args.pred_steps:]
        if args.use_month:
            test_in = test_enc['month'].tolist() + test_enc['cases'].tolist()+test_enc['deaths'].tolist()+test_enc[travelName].values.reshape(-1,order='F').tolist()
            test_out = test_target['month'].tolist() + test_target['cases'].tolist()+test_target['deaths'].tolist()
        else:
            test_in = test_enc['cases'].tolist()+test_enc['deaths'].tolist()+test_enc[travelName].values.reshape(-1,order='F').tolist()
            test_out = test_target['cases'].tolist()+test_target['deaths'].tolist()
        test_set = statecounty + [fip] + [test_target.date.iloc[0]] +test_in + test_out
        testdf = testdf.append(pd.DataFrame(dict(zip(colNames,test_set)),index=[0]),ignore_index=True)
    testdf.reset_index(drop=True,inplace=True)
    testdf.to_csv(f'./useless/Cal_test{encoder_dim}_{args.pred_steps}.csv', index=False)
    print(encoder_dim)
    print(testdf.shape)

if __name__ == '__main__':
    path = 'D:/Desktop/ht/covinet/new'
    os.chdir(path)
    main()


