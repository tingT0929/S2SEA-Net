import pandas as pd
import random
import numpy as np
import tensorflow as tf
import os
import logging
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
import smtplib
from email.mime.text import MIMEText
from email.header import Header


def create_logger(log_dir):
    # os.makedirs(log_dir, exist_ok=True)
    # time_str = time.strftime('%m-%d-%H-%M')
    # log_file = '{}.log'.format(time_str)
    # final_log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '[%(asctime)s] %(message)s'

    file = logging.FileHandler(filename=log_dir, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)
    return logger

def set_seed(seed):
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def get_scaler(dataType, fips, cfg):
    data = pd.read_csv(cfg.data.root + 'covid-travel-new.csv')
    data = data[data['fips'].isin(fips)]
    tempTime = data['date'].unique()
    tempTime.sort()
    data = data[~data['date'].isin(tempTime[-cfg.data.pred_steps:])]
    if 'covid' in dataType:
        df = data.loc[:, ['cases', 'deaths']]
    elif 'travel' in dataType:
        if cfg.data.agg_travel:
            df = pd.DataFrame(columns=['<1', '1-50', '50-250', '>=250'])
            df['<1'] = data.iloc[:, -10]
            df['1-50'] = data.iloc[:, -9:-4].sum(axis=1)
            df['50-250'] = data.iloc[:, -4:-2].sum(axis=1)
            df['>=250'] = data.iloc[:,-2:].sum(axis=1)
        else:
            df = data.iloc[:,-10:]
    else:
        print('DataType is wrong.')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(df)
    return scaler

def transform_data(data, cfg, scaler):
    dataset = data.drop(['StateCounty', 'fips', 'date'], axis = 1)
    colname = list(dataset.columns.values)
    scaler_covid, scaler_travel = scaler
    for i in range(cfg.data.enc_steps):
        dataset.iloc[:,[cfg.data.use_month*cfg.data.enc_steps+ i,(cfg.data.use_month+1)*cfg.data.enc_steps+ i]] = \
            scaler_covid.transform(dataset.iloc[:,[cfg.data.use_month*cfg.data.enc_steps+ i,(cfg.data.use_month+1)*cfg.data.enc_steps+ i]])
        travel_col = [i+(2+cfg.data.use_month)*cfg.data.enc_steps+k*cfg.data.enc_steps for k in range(cfg.model.travel_num)]
        dataset.iloc[:,travel_col] = scaler_travel.transform(dataset.iloc[:,travel_col])

    for i in range(cfg.data.pred_steps):
        dataset.iloc[:,[-2*cfg.data.pred_steps+i,-cfg.data.pred_steps+i]] = scaler_covid.transform(dataset.iloc[:,[-2*cfg.data.pred_steps+i,-cfg.data.pred_steps+i]])

    encoder_inputs = dataset[colname[0:cfg.model.encoder_dim*cfg.data.enc_steps]]
    encoder_inputs = encoder_inputs.values.reshape((-1, cfg.data.enc_steps, cfg.model.encoder_dim), order='F')
    decoder_target = dataset[colname[-cfg.model.decoder_dim*cfg.data.pred_steps:]]
    decoder_target = decoder_target.values.reshape((-1, cfg.data.pred_steps, cfg.model.decoder_dim), order='F')
    decoder_inputs = np.zeros((decoder_target.shape[0], decoder_target.shape[1], cfg.model.decoder_dim + cfg.data.use_month))
    if cfg.data.use_month:
        decoder_inputs[:,1:,0] = dataset.iloc[:,-3*cfg.data.pred_steps:-2*cfg.data.pred_steps-1].values
        decoder_inputs[:,1:,1:] = decoder_target[:,:-1,:]
        decoder_inputs[:,0,:] = encoder_inputs[:,-1,:cfg.model.decoder_dim + cfg.data.use_month]
    else:
        decoder_inputs[:,0,:] = encoder_inputs[:,-1,:cfg.model.decoder_dim + cfg.data.use_month]
        decoder_inputs[:,1:,:] = decoder_target[:,:-1,:]

    if 'Fac' in cfg.network or 'New' in cfg.network:
        factor = pd.read_csv(cfg.data.root+'factor.csv')
        state = factor['state'].unique().tolist()
        #county = factor['StateCounty'].unique().tolist()
        state2id = dict(zip(state, range(len(state))))
        #county2id = dict(zip(county, range(len(county))))
        factor_data = pd.merge(data, factor, on=['StateCounty'], how='left')
        if factor_data.isnull().sum().sum():
            raise ValueError(f'Datasets are not matched.')
        u = [-92, 38.35]
        sigma = [12.21, 5.08]
        factor_data.loc[:, ['LON', 'LAT']] = (factor_data.loc[:, ['LON', 'LAT']] - u) /sigma
        if 'Fac' in cfg.network:
            factor_data = factor_data[['state', 'LON', 'LAT']]
            factor_data['state'] = [state2id[state] for state in factor_data['state'].tolist()]
        #factor_data['StateCounty'] = [county2id[county] for county in factor_data['StateCounty'].tolist()]
        else:
            factor_data = factor_data[['LON', 'LAT']]
        factor_data = factor_data.values
        return encoder_inputs, decoder_inputs, factor_data, decoder_target
    elif 'Seq2' in cfg.network:
        return encoder_inputs[:,:,:2], decoder_inputs, decoder_target 
    return encoder_inputs, decoder_inputs, decoder_target

def get_attentions(model, inputs, type):
    attn1 = Model(inputs=model.input,outputs=model.get_layer(f'{type}').output)
    return attn1.predict(inputs)

def mean_relative_error_covid(y_true, y_pred):
    y_true_tmp = y_true.copy()
    y_true_tmp[y_true_tmp==0] = 1 #smooth
    temp1 = np.abs(y_true[:, 0] - y_pred[:, 0]) / y_true_tmp[:, 0]
    temp2 = np.abs(y_true[:, 1] - y_pred[:, 1]) / y_true_tmp[:, 1]
    temp1 = temp1[~np.isinf(temp1)]
    temp2 = temp2[~np.isinf(temp2)]
    relative_error1 = np.nanmean(temp1, axis=0)
    relative_error2 = np.nanmean(temp2, axis=0)
    return relative_error1, relative_error2

def finishCode(content):
    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText('Code Finished!', 'plain', 'utf-8')   # 邮件内容
    message['From'] = "lawliet0211@163.com"   # 邮件发送者名字
    message['To'] = "309958725@qq.com"   # 邮件接收者名字
    message['Subject'] = Header(content, 'utf-8')   # 邮件主题
    
    try:
        mail = smtplib.SMTP()
        mail.connect("smtp.163.com")   # 连接 163 邮箱
        mail.login("lawliet0211@163.com", "ZKEWSYWZSNZXKCSB")   # 账号和授权码
        mail.sendmail("lawliet0211@163.com", ["309958725@qq.com"], message.as_string())   # 发送账号、接收账号和邮件信息
        print("Email sent successfully!")
    except smtplib.SMTPException:
        print("Error: Unable to send mail.") 
