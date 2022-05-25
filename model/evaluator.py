import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.utils import transform_data, mean_relative_error_covid
from datetime import timedelta
import seaborn as sns
import pandas as pd
import os
import logging
sns.set()

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cases_rmse = 0
        self.deaths_rmse = 0
        self.count = 0
        self.cases_avg = 0
        self.deaths_avg = 0
    
    def update(self, cases, deaths):
        self.cases_rmse += cases
        self.deaths_rmse += deaths
        self.count += 1
        self.cases_avg = self.cases_rmse / self.count
        self.deaths_avg = self.deaths_rmse / self.count

class Evaluator:
    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.trainer = trainer
        self.travelname = ['Number of Trips <1',
            'Number of Trips 1-3',
            'Number of Trips 3-5',
            'Number of Trips 5-10',
            'Number of Trips 10-25',
            'Number of Trips 25-50',
            'Number of Trips 50-100',
            'Number of Trips 100-250',
            'Number of Trips 250-500',
            'Number of Trips >=500'] if not cfg.data.agg_travel else \
                ['Number of Trips <1',
                'Number of Trips 1-50',
                'Number of Trips 50-250',
                'Number of Trips >=250']
    
    def evaluate(self):
        data = pd.read_csv(self.cfg.data.root + 'covid-travel-new.csv')
        states = data.state.unique().tolist()
        plot_states = ['Texas', 'California', 'Arizona', 'New York', 'Florida']
        colNames = ['state','train_cases','train_deaths','test_cases','test_deaths']
        collector = pd.DataFrame(columns=colNames)
        for state in states:
            if state in plot_states:
                self.state_log_dir = self.cfg.save_path + f'{state}/'
                os.makedirs(self.state_log_dir, exist_ok=True)
                with open(self.state_log_dir + 'data.txt', 'w') as f:
                    f.write('-----------------' + state + '---------------\n')
            sub_fips = data[data['state']==state].fips.unique().tolist()
            trainMeter = AverageMeter()
            testMeter = AverageMeter()
            attn_list = []
            plot=False
            if state in plot_states:
                plot = True
            for fip in sub_fips:
                metrics = self.evaluate_county(fip, plot)
                trainMeter.update(metrics[0],metrics[1])
                testMeter.update(metrics[2], metrics[3])
            if 'Attn' in self.cfg.network:
                attn_list.append(metrics[4])
                attn_state = np.mean(attn_list, axis=0)
                if plot:
                    sns.heatmap(attn_state.T,cmap='Blues',yticklabels=self.travelname)
                    plt.savefig(self.state_log_dir + f'{state}_attn.jpg',dpi=200, bbox_inches='tight')
                    plt.close()
            row = [state, trainMeter.cases_avg, trainMeter.deaths_avg, testMeter.cases_avg, testMeter.deaths_avg]
            logging.info(row)
            collector = collector.append(pd.DataFrame(dict(zip(colNames,row)),index=[0]),ignore_index=True)
        collector.loc['Mean_RMSE'] = collector.mean()
        logging.info(['Mean_RMSE'] + collector.mean().tolist())
        collector.reset_index(drop=True,inplace=True)
        collector.to_csv(self.cfg.result_path + f'{self.cfg.network}_{self.cfg.model.encoder_dim}_{self.cfg.data.pred_steps}.csv', index=False)

    def evaluate_county(self, fip, plot=False):
        sub_train = self.trainer.trainDF[self.trainer.trainDF['fips']==fip]
        sub_test = self.trainer.testDF[self.trainer.testDF['fips']==fip]
        stateCountyName = self.trainer.testDF[self.trainer.testDF['fips'] == fip]['StateCounty'].values[0]
        if plot:
            self.county_log_dir = self.state_log_dir + f"{stateCountyName}/"
            os.makedirs(self.county_log_dir, exist_ok=True)
            with open(self.state_log_dir + 'data.txt', 'a') as f:
                f.write('---------' + stateCountyName + '---------\n')
        train_county = transform_data(sub_train, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
        test_county = transform_data(sub_test, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
        train_out = self.trainer.network.train_model.predict(train_county[:-1])
        train_predict = np.zeros((train_out.shape[0]+self.cfg.data.pred_steps-1,self.cfg.model.decoder_dim))
        for i in range(train_out.shape[0]-1):
            train_predict[i,:] = train_out[i][0]
        train_predict[-self.cfg.data.pred_steps:] = train_out[-1]
        train_predict = self.trainer.scaler_covid.inverse_transform(train_predict).astype(np.int32)
        train_target = np.zeros((train_county[-1].shape[0]+self.cfg.data.pred_steps-1,2))
        for i in range(train_out.shape[0]-1):
            train_target[i,:] = train_county[-1][i][0]
        train_target[-self.cfg.data.pred_steps:] = train_county[-1][-1]
        train_target = self.trainer.scaler_covid.inverse_transform(train_target).astype(np.int32)
        train_cases_rmse, train_deaths_rmse = mean_squared_error(train_target, train_predict, multioutput='raw_values', squared=False)
        test_target = self.trainer.scaler_covid.inverse_transform(test_county[-1].squeeze(0)).astype(np.int32)
        net_output = self.trainer.network.predict(test_county)
        test_predict = net_output[0]
        test_predict = self.trainer.scaler_covid.inverse_transform(test_predict.squeeze(0)).astype(np.int32)
        test_cases_rmse, test_deaths_rmse = mean_squared_error(test_target, test_predict, multioutput='raw_values', squared=False)
        if plot:
            with open(self.state_log_dir + 'data.txt', 'a') as f:
                f.write(f'Train Cases RMSE: {train_cases_rmse:.2f}, Train Deaths RMSE: {train_deaths_rmse:.2f}\n')
                f.write(f'Test Cases RMSE: {test_cases_rmse:.2f}, Test Deaths RMSE: {test_deaths_rmse:.2f}\n')
            origin_data = np.vstack([train_target, test_target])
            test_pred = np.vstack([train_predict[-1],test_predict])
            plt.subplot(121)
            plt.plot(range(origin_data.shape[0]), origin_data[:,0])
            plt.plot(range(train_predict.shape[0]), train_predict[:,0], color='red')
            plt.plot(range(train_predict.shape[0]-1, origin_data.shape[0]), test_pred[:,0], color='green')
            plt.title(stateCountyName + ' Cases')
            plt.legend(['Ground Truth', 'Train Predicted','Test Predicted'])
            plt.subplot(122)
            plt.plot(range(self.cfg.data.pred_steps), test_target[:,0])
            plt.plot(range(self.cfg.data.pred_steps), test_predict[:,0], color='green')
            plt.title(stateCountyName + ' Predicted Cases')
            plt.legend(['Truth', 'Predicted'])
            plt.savefig(self.county_log_dir + stateCountyName + '_cases.jpg',dpi=200, bbox_inches='tight')
            plt.close()
            plt.subplot(121)
            plt.plot(range(origin_data.shape[0]), origin_data[:,1])
            plt.plot(range(train_predict.shape[0]), train_predict[:,1], color='red')
            plt.plot(range(train_predict.shape[0]-1, origin_data.shape[0]), test_pred[:,1], color='green')
            plt.title(stateCountyName + ' Deaths')
            plt.legend(['Ground Truth', 'Train Predicted','Test Predicted'])
            plt.subplot(122)
            plt.plot(range(self.cfg.data.pred_steps), test_target[:,1])
            plt.plot(range(self.cfg.data.pred_steps), test_predict[:,1], color='green')
            plt.title(stateCountyName + ' Predicted Deaths')
            plt.legend(['Truth', 'Predicted'])
            plt.savefig(self.county_log_dir + stateCountyName + '_deaths.jpg',dpi=200, bbox_inches='tight')
            plt.close()
        metrics = [train_cases_rmse, train_deaths_rmse, test_cases_rmse, test_deaths_rmse]
        if len(net_output) == 2:
            attn = net_output[-1].squeeze(0)
            metrics.append(attn)
            if plot:
                sns.heatmap(attn.T, cmap='Blues',yticklabels=self.travelname)
                plt.savefig(self.county_log_dir + stateCountyName + '_attn.jpg',dpi=200, bbox_inches='tight')
                plt.close()

        return metrics

    def get_attentions(self, name):
        data_list = []
        data = pd.read_csv(self.cfg.data.root + 'covid-travel-new.csv')
        states = data.state.unique().tolist()
        for state in states:
            sub_fips = data[data['state']==state].fips.unique().tolist()
            for fip in sub_fips:
                traindf = self.trainer.trainDF[self.trainer.trainDF['fips']==fip]
                testdf = self.trainer.testDF[self.trainer.testDF['fips']==fip]
                stateCountyName = self.trainer.testDF[self.trainer.testDF['fips'] == fip]['StateCounty'].values[0]
                traindf.loc[:,'date'] = pd.to_datetime(traindf.loc[:,'date'])
                cases = traindf['cases_30'].tolist() + [traindf.iloc[-1][f'cases_{i+1}'] for i in range(1,self.cfg.data.enc_steps)]
                deaths = traindf['deaths_30'].tolist() + [traindf.iloc[-1][f'deaths_{i+1}'] for i in range(1,self.cfg.data.enc_steps)]
                period = np.hstack([traindf['date'].values, pd.Series([traindf['date'].iloc[-1] + timedelta(i) for i in range(1, self.cfg.data.enc_steps)]).values])
                train_county = transform_data(traindf, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
                #test_county = transform_data(sub_test, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
                if 'Fac' in self.cfg.network:
                    _, attn = self.trainer.network.encoder_model.predict([train_county[0]]+[train_county[2]])
                else:
                    _, attn = self.trainer.network.encoder_model.predict(train_county[0])
                #attn0 = np.vstack([attn[:-1,0,:], attn[-1,:,:]])
                b = dict()
                for i in range(attn.shape[0]+attn.shape[1] - 1):
                    b[i] = []
                for i in range(attn.shape[0]):
                    temp = attn[i]
                    for j in range(30):
                        b[i + j].append(temp[j])
                for i in range(attn.shape[0]+attn.shape[1] - 1):
                    b[i] = np.mean(b[i], axis=0)
                attn0 = np.vstack(list(b.values())) 
                size = len(cases)
                data_dict = {'State':[state]*size,'StateCounty':[stateCountyName]*size, 'date':period, 'cases':cases,
                            'deaths':deaths}
                data_dict.update(dict(zip(self.travelname,attn0.T)))
                data_fip = pd.DataFrame(data_dict)
                data_list.append(data_fip)

        dataDF = pd.concat(data_list)
        dataDF.reset_index(drop=True, inplace=True)
        dataDF.to_csv(f'{self.cfg.result_path}/{self.cfg.network}_{self.cfg.model.encoder_dim}_{self.cfg.data.pred_steps}_attention_{name}.csv', index=False)

    def get_pred(self, fip):
        sub_test = self.trainer.testDF[self.trainer.testDF['fips']==fip]
        stateCountyName = self.trainer.testDF[self.trainer.testDF['fips'] == fip]['StateCounty'].values[0]
        test_county = transform_data(sub_test, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
        test_target = self.trainer.scaler_covid.inverse_transform(test_county[-1].squeeze(0)).astype(np.int32)
        net_output = self.trainer.network.predict(test_county)
        test_predict = net_output[0]
        test_predict = self.trainer.scaler_covid.inverse_transform(test_predict.squeeze(0)).astype(np.int32)

        if len(net_output) == 2:
            attn = net_output[-1].squeeze(0)
            np.save(self.cfg.result_path + f'{stateCountyName}_attn{self.cfg.data.pred_steps}.npy', attn)

        np.save(self.cfg.result_path + f'{stateCountyName}_target{self.cfg.data.pred_steps}.npy', test_target)
        np.save(self.cfg.result_path + f'{stateCountyName}_predict{self.cfg.data.pred_steps}.npy', test_predict)
        

    def evaluate_cum(self):
        data = pd.read_csv(self.cfg.data.root + 'covid-travel-new.csv')
        states = data.state.unique().tolist()
        plot_states = ['Texas', 'California', 'Arizona', 'New York', 'Florida']
        colNames = ['state','train_cases','train_deaths','test_cases','test_deaths']
        collector = pd.DataFrame(columns=colNames)
        for state in states:
            if state in plot_states:
                self.state_log_dir = self.cfg.save_path + f'{state}/'
                os.makedirs(self.state_log_dir, exist_ok=True)
                with open(self.state_log_dir + 'data_cum.txt', 'w') as f:
                    f.write('-----------------' + state + '---------------\n')
            sub_fips = data[data['state']==state].fips.unique().tolist()
            trainMeter = AverageMeter()
            testMeter = AverageMeter()
            attn_list = []
            plot=False
            if state in plot_states:
                plot = True
            for fip in sub_fips:
                metrics = self.evaluate_county_cum(fip, plot)
                trainMeter.update(metrics[0],metrics[1])
                testMeter.update(metrics[2], metrics[3])
            # if 'Attn' in self.cfg.network:
            #     attn_list.append(metrics[4])
            #     attn_state = np.mean(attn_list, axis=0)
            #     if plot:
            #         sns.heatmap(attn_state,cmap='Blues')
            #         plt.savefig(self.state_log_dir + f'{state}_attn.jpg',dpi=200, bbox_inches='tight')
            #         plt.close()
            row = [state, trainMeter.cases_avg, trainMeter.deaths_avg, testMeter.cases_avg, testMeter.deaths_avg]
            logging.info(row)
            collector = collector.append(pd.DataFrame(dict(zip(colNames,row)),index=[0]),ignore_index=True)
        collector.loc['Mean_mre'] = collector.mean()
        logging.info(['Mean_mre'] + collector.mean().tolist())
        collector.reset_index(drop=True,inplace=True)
        collector.to_csv(self.cfg.result_path + f'{self.cfg.network}_{self.cfg.model.encoder_dim}_{self.cfg.data.pred_steps}_cum.csv', index=False)

    def evaluate_county_cum(self, fip, plot=False):
        sub_train = self.trainer.trainDF[self.trainer.trainDF['fips']==fip]
        sub_test = self.trainer.testDF[self.trainer.testDF['fips']==fip]
        stateCountyName = self.trainer.testDF[self.trainer.testDF['fips'] == fip]['StateCounty'].values[0]
        if plot:
            self.county_log_dir = self.state_log_dir + f"{stateCountyName}/"
            os.makedirs(self.county_log_dir, exist_ok=True)
            with open(self.state_log_dir + 'data_cum.txt', 'a') as f:
                f.write('---------' + stateCountyName + '---------\n')
        train_county = transform_data(sub_train, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
        test_county = transform_data(sub_test, self.cfg, [self.trainer.scaler_covid, self.trainer.scaler_travel])
        train_out = self.trainer.network.train_model.predict(train_county[:-1])
        train_predict = np.zeros((train_out.shape[0]+self.cfg.data.pred_steps-1,self.cfg.model.decoder_dim))
        for i in range(train_out.shape[0]-1):
            train_predict[i,:] = train_out[i][0]
        train_predict[-self.cfg.data.pred_steps:] = train_out[-1]
        train_predict = self.trainer.scaler_covid.inverse_transform(train_predict).astype(np.int32)
        train_target = np.zeros((train_county[-1].shape[0]+self.cfg.data.pred_steps-1,2))
        for i in range(train_out.shape[0]-1):
            train_target[i,:] = train_county[-1][i][0]
        train_target[-self.cfg.data.pred_steps:] = train_county[-1][-1]
        train_target = self.trainer.scaler_covid.inverse_transform(train_target).astype(np.int32)
        train_target_cum = np.cumsum(train_target, axis=0)
        train_predict_cum = np.cumsum(train_predict, axis=0)
        train_cases_mre, train_deaths_mre = mean_relative_error_covid(train_target_cum, train_predict_cum)
        test_target = self.trainer.scaler_covid.inverse_transform(test_county[-1].squeeze(0)).astype(np.int32)
        net_output = self.trainer.network.predict(test_county)
        test_predict = net_output[0]
        test_predict = self.trainer.scaler_covid.inverse_transform(test_predict.squeeze(0)).astype(np.int32)
        test_target[0] = test_target[0] + train_target_cum[-1]
        test_predict[0] = test_predict[0] + train_predict_cum[-1]
        test_target_cum = np.cumsum(test_target, axis=0) 
        test_predict_cum = np.cumsum(test_predict, axis=0)
        test_cases_mre, test_deaths_mre = mean_relative_error_covid(test_target_cum, test_predict_cum)
        if plot:
            with open(self.state_log_dir + 'data_cum.txt', 'a') as f:
                f.write(f'Train Cases RMSE: {train_cases_mre:.2f}, Train Deaths RMSE: {train_deaths_mre:.2f}\n')
                f.write(f'Test Cases RMSE: {test_cases_mre:.2f}, Test Deaths RMSE: {test_deaths_mre:.2f}\n')
            origin_data = np.vstack([train_target_cum, test_target_cum])
            test_pred = np.vstack([train_predict_cum[-1],test_predict_cum])
            plt.subplot(121)
            plt.plot(range(origin_data.shape[0]), origin_data[:,0])
            plt.plot(range(train_predict.shape[0]), train_predict_cum[:,0], color='red')
            plt.plot(range(train_predict.shape[0]-1, origin_data.shape[0]), test_pred[:,0], color='green')
            plt.title(stateCountyName + ' Cases')
            plt.legend(['Ground Truth', 'Train Predicted','Test Predicted'])
            plt.subplot(122)
            plt.plot(range(self.cfg.data.pred_steps), test_target_cum[:,0])
            plt.plot(range(self.cfg.data.pred_steps), test_predict_cum[:,0], color='green')
            plt.title(stateCountyName + ' Predicted Cases')
            plt.legend(['Truth', 'Predicted'])
            plt.savefig(self.county_log_dir + stateCountyName + '_cum_cases.jpg',dpi=200, bbox_inches='tight')
            plt.close()
            plt.subplot(121)
            plt.plot(range(origin_data.shape[0]), origin_data[:,1])
            plt.plot(range(train_predict.shape[0]), train_predict_cum[:,1], color='red')
            plt.plot(range(train_predict.shape[0]-1, origin_data.shape[0]), test_pred[:,1], color='green')
            plt.title(stateCountyName + ' Deaths')
            plt.legend(['Ground Truth', 'Train Predicted','Test Predicted'])
            plt.subplot(122)
            plt.plot(range(self.cfg.data.pred_steps), test_target_cum[:,1])
            plt.plot(range(self.cfg.data.pred_steps), test_predict_cum[:,1], color='green')
            plt.title(stateCountyName + ' Predicted Deaths')
            plt.legend(['Truth', 'Predicted'])
            plt.savefig(self.county_log_dir + stateCountyName + '_cum_deaths.jpg',dpi=200, bbox_inches='tight')
            plt.close()
        metrics = [train_cases_mre, train_deaths_mre, test_cases_mre, test_deaths_mre]
        # if len(net_output) == 2:
        #     attn = net_output[-1].squeeze(0)
        #     metrics.append(attn)
        #     if plot:
        #         sns.heatmap(attn, cmap='Blues')
        #         plt.savefig(self.county_log_dir + stateCountyName + '_attn.jpg',dpi=200, bbox_inches='tight')
        #         plt.close()

        return metrics