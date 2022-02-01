import flask
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import keras
import tushare as ts
import datetime
from tensorflow.keras.layers import Dropout, Dense, GRU, LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from flask_cors import CORS



ts.set_token('')                #自己去tushare申请token！！！！！！！！！！！
pro = ts.pro_api()              #初始化tushare
host='0.0.0.0'
save_path_head = "D:/model/api_saved/"

# 实例化 flask 
app = flask.Flask(__name__)
CORS(app)

# 加载模型
global graph
#graph=tf.reset_default_graph()
#model = load_model('D:/model/1')

@app.route('/')
def index():
    f = open('D:\学习\毕业设计\接口\\readme.md',"r")   #设置文件对象
    str_readme = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close() 
    return str_readme

@app.route('/hello', methods=["GET","POST"])
def hello():
    return 'Hello World'

# 将预测函数定义为一个路由
@app.route("/predict", methods=["GET","POST"])
def predict():
    step_length = 180
    data = {"success": False}
    params = flask.request.json  
    if (params != None):
        try:
            days = int(params['days'])
        except:
            days = 1
        if (days!=1 and days != 3 and days != 5):
            return "输入的days只能为1,3,5"
        now = datetime.datetime.now()
        delta = datetime.timedelta(days=2000)
        n_days = now - delta
        start_date = n_days.strftime('%Y%m%d')
        end_date = now.strftime('%Y%m%d')
        # df = pro.index_basic(market = 'SSE')
        # print(df)
        df = pro.index_daily(start_date=start_date, end_date=end_date, ts_code="000001.SH")
        print(df)
        
        print(df["close"])
        print(df["close"].values)
        print(df["close"].values.astype('float64'))
        close = df["close"].values.astype('float64')
        close = close[::-1]
        print(close[1330])
        print(close)
        close = close.reshape(-1,1)
        print(close)
        scaler_close = MinMaxScaler(feature_range=(0,1))
        close = scaler_close.fit_transform(close)
        
        vol = df["vol"].values.astype('float64')
        vol = vol[::-1]
        vol = vol.reshape(-1,1)
        scaler_vol = MinMaxScaler(feature_range=(0,1))
        vol = scaler_vol.fit_transform(vol)
        dataset = np.concatenate((close,vol),axis=1)

        # open_data = df["open"].values.astype('float64')
        # open_data = open_data[::-1]
        # open_data = open_data.reshape(-1,1)
        # scaler_open = MinMaxScaler(feature_range=(0,1))
        # open_data = scaler_open.fit_transform(open_data)
        # dataset = np.concatenate((dataset,open_data),axis=1)

        # high = df["high"].values.astype('float64')
        # high = high[::-1]
        # high = high.reshape(-1,1)
        # scaler_high = MinMaxScaler(feature_range=(0,1))
        # high = scaler_high.fit_transform(high)
        # dataset = np.concatenate((dataset,high),axis=1)

        # low = df["low"].values.astype('float64')
        # low = low[::-1]
        # low = low.reshape(-1,1)
        # scaler_low = MinMaxScaler(feature_range=(0,1))
        # low = scaler_low.fit_transform(low)
        # dataset = np.concatenate((dataset,low),axis=1)
        

        test_size = 210
        train_size = len(dataset) - 210
        test_set = dataset[train_size:len(dataset),:]


        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_test_set = test_set.copy()
        real_stock_price = scaler_close.inverse_transform(real_test_set[:,0].reshape(-1,1))     #因为判断涨跌正确率需要前一天数据，所以比预测多一天，下面会分别处理
        real_stock_price = real_stock_price[step_length-1:]
        real_stock_price_show=[]
        for i in real_stock_price:
            real_stock_price_show.append(i[0])

        #处理预测数据
        predicted_stock_price_final = []
        pre_set = []
        for i in range(step_length, len(test_set)):
            pre_set.append(test_set[i - step_length : i,:])

        test_set_use = test_set.copy()

        if (days == 1):
            model = tf.saved_model.load("D:/model/300/180_1day/gru_lstm/80gru_100gru_sigmoid_mse_adam_b64_100_1")
            #预测
            # count = 0
            # predict_list = []
            # for i in range(0,len(pre_set)):
            #     pre_set[i] = np.expand_dims(pre_set[i], axis=0)
            #     predicted_stock_price_tmp = model(pre_set[i])
            #     predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
            #     predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            # predict_arr = np.array(predict_list)
            # predict_next_day = predict_arr[-1,:]

            # deviation = 0.0
            # for i in range(0,len(real_stock_price_show)-2):
            #     deviation += abs(real_stock_price_show[i+1] - predicted_stock_price_final[i]) / real_stock_price_show[i+1]
            # deviation /= (len(real_stock_price_show)-1)
            # acc = 1 - deviation
            # print("准确率：",acc*100,"%")

            # acc_longOrShort = 0.0
            # right_count = 0
            # for i in range(0,len(real_stock_price_show)-2):
            #     if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predicted_stock_price_final[i] - real_stock_price_show[i]) >=0 :
            #         right_count +=1
            # acc_longOrShort = right_count / (len(real_stock_price_show)-2)
            # print("涨跌正确率：",acc_longOrShort*100,"%")

            # acc_profit = 1.0
            # buy_count = 0
            # for i in range(0,len(real_stock_price_show)-2):
            #     if predicted_stock_price_final[i] - real_stock_price_show[i]>=0 :
            #         acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
            #         buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])  #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_1day = predict_arr[-1,:]

            deviation = 0.0
            for i in range(0,len(real_stock_price_show)-2):
                deviation += abs(real_stock_price_show[i+1] - predict_arr[i]) / real_stock_price_show[i+1]
            deviation /= (len(real_stock_price_show)-1)
            acc = 1 - deviation
            print("准确率：",acc*100,"%")

            acc_longOrShort = 0.0
            right_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predict_arr[i] - real_stock_price_show[i]) >=0 :
                    right_count +=1
            acc_longOrShort = right_count / (len(real_stock_price_show)-2)
            print("涨跌正确率：",acc_longOrShort*100,"%")

            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predict_arr[i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(buy_count)
            show = str(predict_next_1day)
            show = show[1:-1]
            print(show)
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"


        elif(days == 3):
            #预测
            model = tf.saved_model.load("D:/model/300/180_3day/gru_lstm/80gru_100gru_sigmoid_mse_sdg_b16_100_3")
            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_3day = predict_arr[-1,:]
            #test_set_use = test_set_use[step_length:-1,].

            #predicted_stock_price = np.zeros((3),dtype= np.float64)
            predicted_stock_price_list = []
            for i in range(0,3):
                predicted_stock_price_list.append(predict_arr[:,i])
            predicted_stock_price = np.array(predicted_stock_price_list)


            deviation = np.zeros((3),dtype= np.float64)
            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,3):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                        deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

            acc = np.zeros((3),dtype= np.float64)
            
            for i in range(0,3):
                deviation[i] /= (len(real_stock_price_show)-1-i)
                acc[i] = 1 -deviation[i]

            #三天的涨跌准确率
            acc_longOrShort = np.zeros((3),dtype= np.float64)
            right_count = np.zeros((3),dtype= np.float64)

            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,3):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                        if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                            right_count[j] += 1

            for i in range(0,3):
                acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



            #三天的收益率
            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(predict_next_3day)
            # show = str(predict_next_3day)
            # print(show)
            show = ''.join(str(i)+"," for i in predict_next_3day)
            show = show[:-1]
            print(show)
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"

        elif(days == 5):
            #预测
            model = tf.saved_model.load("D:/model/300/180_5day/gru_lstm/80gru_100gru_sigmoid_mse_sdg_b16_100_5")
            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_5day = predict_arr[-1,:]
            #test_set_use = test_set_use[step_length:-1,].

            #predicted_stock_price = np.zeros((3),dtype= np.float64)
            predicted_stock_price_list = []
            for i in range(0,days):
                predicted_stock_price_list.append(predict_arr[:,i])
            predicted_stock_price = np.array(predicted_stock_price_list)


            deviation = np.zeros((days),dtype= np.float64)
            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,days):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                        deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

            acc = np.zeros((days),dtype= np.float64)
            
            for i in range(0,days):
                deviation[i] /= (len(real_stock_price_show)-1-i)
                acc[i] = 1 -deviation[i]

            #五天的涨跌准确率
            acc_longOrShort = np.zeros((days),dtype= np.float64)
            right_count = np.zeros((days),dtype= np.float64)

            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,days):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                        if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                            right_count[j] += 1

            for i in range(0,days):
                acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



            #五天的收益率
            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(buy_count)
            show = ''.join(str(i)+"," for i in predict_next_5day)
            show = show[:-1]
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"
            
        trade_date = df["trade_date"].values.astype('int')
        trade_date = trade_date[::-1]
        trade_date = trade_date.reshape(-1,1)
        before_date = ''.join(str(i[0])+"," for i in trade_date[-10:])
        before_date = before_date[:-1]

        real_10days = ''.join(str(i)+"," for i in real_stock_price_show[-10:])
        real_10days = real_10days[:-1]

        data["before_date"] = before_date
        data["real_10days"] = real_10days
        data["success"] = True
            # 返回Json格式的响应
        # model = tf.saved_model.load("D:/model/300/180_1day/16_32_gru_sigmoid_huber_sdg_b32")
        # #预测数据 并反归一化
        # for i in range(0,len(pre_set)):
        #     pre_set[i] = np.expand_dims(pre_set[i], axis=0)
        #     tmp = tf.convert_to_tensor(pre_set[i])
        #     print(tmp)
        #     predicted_stock_price_tmp = model(tmp)
        #     predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
        #     test_set_use[i+step_length][0] = predicted_stock_price_tmp[0]

        # test_set_use = test_set_use[step_length:,]
        # predicted_stock_price_final = test_set_use[:,0]


        # print("real",len(real_stock_price_show))
        # print("pre",len(predicted_stock_price_final))

        # #数据误差的准确率
        # deviation = 0.0
        # for i in range(0,len(real_stock_price_show)-2):
        #     deviation += abs(real_stock_price_show[i+1] - predicted_stock_price_final[i]) / real_stock_price_show[i+1]

        # deviation /= (len(real_stock_price_show)-1)
        # acc = 1 - deviation
        # print("准确率：",acc*100,"%")

        # #涨跌判断的正确率
        # acc_longOrShort = 0.0
        # right_count = 0
        # for i in range(0,len(real_stock_price_show)-2):
        #     if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predicted_stock_price_final[i] - real_stock_price_show[i]) >=0 :
        #         right_count +=1

        # acc_longOrShort = right_count / (len(real_stock_price_show)-1)
        # print("涨跌正确率：",acc_longOrShort*100,"%")

        # acc_profit = 1.0
        # for i in range(0,len(real_stock_price_show)-2):
        #     if predicted_stock_price_final[i] - real_stock_price_show[i]>=0 :
        #         acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
        #     else :
        #         acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)
        # acc_profit -= 1
        # print("盈亏率(可空头))：",acc_profit*100,"%")

        # show = str(predicted_stock_price_final[len(predicted_stock_price_final)-1])
        # data["该模型准确率"] = str(acc*100)+"%"
        # data["该模型涨跌正确率"] = str(acc_longOrShort*100)+"%"
        # data["该模型盈亏率(可空头)"] = str(acc_profit*100)+"%"
        # data["predict"] = show
        # data["success"] = True


    return flask.jsonify(data)    


@app.route('/trainAndPredict', methods=["GET","POST"])
def trainAndPredict():
    data = {"success": False}
    params = flask.request.json  
    if (params != None):
        if (params['code']!= None):
            echos = 300
            batch_size = 64
            try:
                echos = int(params['echos'])
            except:
                echos = 300

            try:
                batch_size = int(params['batch_size'])
            except KeyError:
                batch_size = 64
            
            try:
                days = int(params['days'])
            except:
                days = 1
            if (days!=1 and days != 3 and days != 5):
                return "输入的days只能为1,3,5"
            ts_code = params['code']
            step_length = 60
            now = datetime.datetime.now()
            delta = datetime.timedelta(days=2000)
            n_days = now - delta
            start_date = n_days.strftime('%Y%m%d')
            end_date = now.strftime('%Y%m%d')


            df = pro.daily(start_date=start_date, end_date=end_date, ts_code=ts_code)
            print(df)
            close = df["close"].values.astype('float64')
            close = close[::-1]
            close = close.reshape(-1,1)
            scaler_close = MinMaxScaler(feature_range=(0,1))
            close = scaler_close.fit_transform(close)
            
            vol = df["vol"].values.astype('float64')
            vol = vol[::-1]
            vol = vol.reshape(-1,1)
            scaler_vol = MinMaxScaler(feature_range=(0,1))
            vol = scaler_vol.fit_transform(vol)
            dataset = np.concatenate((close,vol),axis=1)

            open_data = df["open"].values.astype('float64')
            open_data = open_data[::-1]
            open_data = open_data.reshape(-1,1)
            scaler_open = MinMaxScaler(feature_range=(0,1))
            open_data = scaler_open.fit_transform(open_data)
            dataset = np.concatenate((dataset,open_data),axis=1)

            high = df["high"].values.astype('float64')
            high = high[::-1]
            high = high.reshape(-1,1)
            scaler_high = MinMaxScaler(feature_range=(0,1))
            high = scaler_high.fit_transform(high)
            dataset = np.concatenate((dataset,high),axis=1)

            low = df["low"].values.astype('float64')
            low = low[::-1]
            low = low.reshape(-1,1)
            scaler_low = MinMaxScaler(feature_range=(0,1))
            low = scaler_low.fit_transform(low)
            dataset = np.concatenate((dataset,low),axis=1)


            test_size = 90
            train_size = len(dataset) - 90
            train_set = dataset[0:train_size,:]
            test_set = dataset[train_size:len(dataset),:]
            x_train = []
            y_train = []

            for i in range(step_length, len(train_set)-days):
                x_train.append(train_set[i - step_length:i, :])
                y_train.append(train_set[i:i+days, 0])
            # 对训练集进行打乱
            np.random.seed(7)
            np.random.shuffle(x_train)
            np.random.seed(7)
            np.random.shuffle(y_train)
            tf.random.set_seed(7)

            x_train, y_train = np.array(x_train), np.array(y_train)
            y_train = np.expand_dims(y_train, axis=2)
            x_train_tf = tf.convert_to_tensor(x_train)
            y_train_tf = tf.convert_to_tensor(y_train)
            #y_train_tf = np.expand_dims(y_train_tf, axis=1)

            x_test = []
            y_test = []
            # 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
            for i in range(step_length, len(test_set)-days):
                x_test.append(test_set[i - step_length:i,:])
                y_test.append(test_set[i:i+days, 0])
            # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
            x_test, y_test = np.array(x_test), np.array(y_test)
            y_test = np.expand_dims(y_test, axis=2)

            model = tf.keras.Sequential([
                # GRU(32, return_sequences=True, activation='tanh'),
                # Dropout(0.2),
                # GRU(32, return_sequences=True, activation='tanh'),
                # Dropout(0.2),
                GRU(32 , activation='tanh'),
                Dropout(0.2),
                Dense(days)
                
            ])
            opt = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=opt,
                        loss='mean_absolute_error'
                        )  # 损失函数用均方误差


            checkpoint_save_path = "D:\checkpoint\stock.ckpt"
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                            save_weights_only=True,
                                                            save_best_only=True,
                                                            monitor='val_loss')
            history = model.fit(x_train_tf, y_train_tf, batch_size=batch_size, epochs=echos, validation_split=0.8, validation_freq=1,
                                callbacks=[cp_callback])

            #反归一化成原始数据，一会作比较
            real_test_set = test_set.copy()
            real_stock_price = scaler_close.inverse_transform(real_test_set[:,0].reshape(-1,1))     #因为判断涨跌正确率需要前一天数据，所以比预测多一天，下面会分别处理
            real_stock_price = real_stock_price[step_length-1:]
            real_stock_price_show=[]
            for i in real_stock_price:
                real_stock_price_show.append(i[0])

            #预测数据准备
            predicted_stock_price_final = []
            pre_set = []
            
            for i in range(step_length, len(test_set)):
                pre_set.append(test_set[i - step_length : i,:])
            pre_set_arr = np.array(pre_set)
            test_set_use = test_set.copy()

            if (days == 1):
                #预测
                count = 0
                for i in range(0,len(pre_set)):
                    pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                    predicted_stock_price_tmp = model.predict(pre_set[i])
                    predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                    test_set_use[i+step_length][0] = predicted_stock_price_tmp[0]   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

                predict_next_day = test_set_use[-1:,]
                test_set_use = test_set_use[step_length:-1,]
                predicted_stock_price_final = test_set_use[:,0]

                deviation = 0.0
                for i in range(0,len(real_stock_price_show)-2):
                    deviation += abs(real_stock_price_show[i+1] - predicted_stock_price_final[i]) / real_stock_price_show[i+1]
                deviation /= (len(real_stock_price_show)-1)
                acc = 1 - deviation
                print("准确率：",acc*100,"%")

                acc_longOrShort = 0.0
                right_count = 0
                for i in range(0,len(real_stock_price_show)-2):
                    if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predicted_stock_price_final[i] - real_stock_price_show[i]) >=0 :
                        right_count +=1
                acc_longOrShort = right_count / (len(real_stock_price_show)-2)
                print("涨跌正确率：",acc_longOrShort*100,"%")

                acc_profit = 1.0
                buy_count = 0
                for i in range(0,len(real_stock_price_show)-2):
                    if predicted_stock_price_final[i] - real_stock_price_show[i]>=0 :
                        acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                        buy_count+=1
                # else :

                    #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

                acc_profit -= 1
                print("盈亏率：",acc_profit*100,"%")
                show = str(predict_next_day[0][0])
                

                data["acc"] = str(acc)
                data["predict"] = show
                data["acc_longOrShort"] = str(acc_longOrShort)
                data["该模型前29天买入次数"] = str(buy_count)+"次"
                data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"


            elif(days == 3):
                #预测
                count = 0
                predict_list = []
                for i in range(0,len(pre_set)):
                    pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                    predicted_stock_price_tmp = model.predict(pre_set[i])
                    predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                    predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

                predict_arr = np.array(predict_list)
                predict_next_3day = predict_arr[-1,:]
                #test_set_use = test_set_use[step_length:-1,].

                #predicted_stock_price = np.zeros((3),dtype= np.float64)
                predicted_stock_price_list = []
                for i in range(0,3):
                    predicted_stock_price_list.append(predict_arr[:,i])
                predicted_stock_price = np.array(predicted_stock_price_list)


                deviation = np.zeros((3),dtype= np.float64)
                for i in range(0,len(real_stock_price_show)-1):
                    for j in range(0,3):
                        if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                            deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

                acc = np.zeros((3),dtype= np.float64)
                
                for i in range(0,3):
                    deviation[i] /= (len(real_stock_price_show)-1-i)
                    acc[i] = 1 -deviation[i]

                #三天的涨跌准确率
                acc_longOrShort = np.zeros((3),dtype= np.float64)
                right_count = np.zeros((3),dtype= np.float64)

                for i in range(0,len(real_stock_price_show)-1):
                    for j in range(0,3):
                        if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                            if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                                right_count[j] += 1

                for i in range(0,3):
                    acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



                #三天的收益率
                acc_profit = 1.0
                buy_count = 0
                for i in range(0,len(real_stock_price_show)-2):
                    if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                        acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                        buy_count+=1
                # else :

                    #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

                acc_profit -= 1
                print("盈亏率：",acc_profit*100,"%")
                print(predict_next_3day)
                # show = str(predict_next_3day)
                # print(show)
                show = ''.join(str(i)+"," for i in predict_next_3day)
                show = show[:-1]
                print(show)
                data["acc"] = str(acc)
                data["predict"] = show
                data["acc_longOrShort"] = str(acc_longOrShort)
                data["该模型前29天买入次数"] = str(buy_count)+"次"
                data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"


            elif(days == 5):
                #预测
                count = 0
                predict_list = []
                for i in range(0,len(pre_set)):
                    pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                    predicted_stock_price_tmp = model.predict(pre_set[i])
                    predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                    predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

                predict_arr = np.array(predict_list)
                predict_next_5day = predict_arr[-1,:]
                #test_set_use = test_set_use[step_length:-1,].

                #predicted_stock_price = np.zeros((3),dtype= np.float64)
                predicted_stock_price_list = []
                for i in range(0,days):
                    predicted_stock_price_list.append(predict_arr[:,i])
                predicted_stock_price = np.array(predicted_stock_price_list)


                deviation = np.zeros((days),dtype= np.float64)
                for i in range(0,len(real_stock_price_show)-1):
                    for j in range(0,days):
                        if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                            deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

                acc = np.zeros((days),dtype= np.float64)
                
                for i in range(0,days):
                    deviation[i] /= (len(real_stock_price_show)-1-i)
                    acc[i] = 1 -deviation[i]

                #五天的涨跌准确率
                acc_longOrShort = np.zeros((days),dtype= np.float64)
                right_count = np.zeros((days),dtype= np.float64)

                for i in range(0,len(real_stock_price_show)-1):
                    for j in range(0,days):
                        if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                            if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                                right_count[j] += 1

                for i in range(0,days):
                    acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



                #五天的收益率
                acc_profit = 1.0
                buy_count = 0
                for i in range(0,len(real_stock_price_show)-2):
                    if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                        acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                        buy_count+=1
                # else :

                    #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

                acc_profit -= 1
                print("盈亏率：",acc_profit*100,"%")

                show = ''.join(str(i)+"," for i in predict_next_5day)
                show = show[:-1]
                data["acc"] = str(acc)
                data["predict"] = show
                data["acc_longOrShort"] = str(acc_longOrShort)
                data["该模型前29天买入次数"] = str(buy_count)+"次"
                data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"
            
            trade_date = df["trade_date"].values.astype('int')
            trade_date = trade_date[::-1]
            trade_date = trade_date.reshape(-1,1)
            before_date = ''.join(str(i[0])+"," for i in trade_date[-10:])
            before_date = before_date[:-1]

            real_10days = ''.join(str(i)+"," for i in real_stock_price_show[-10:])
            real_10days = real_10days[:-1]

            data["before_date"] = before_date
            data["real_10days"] = real_10days
            data["success"] = True
            # 返回Json格式的响应
            model_save_path = save_path_head+ts_code[:6]+ts_code[7:]+"_"+str(echos)+"_"+str(batch_size)+"_"+now.strftime('%Y%m%d%H%M%S')+"_"+str(days)
            tf.saved_model.save(model, model_save_path)
            data["模型路径值(下次可通过此值调用本次训练的模型)"] = model_save_path
            return flask.jsonify(data)  
        else:
            return '请在RequestBody中加入您要选择训练的公司编号(如：{"code":"601998.SH"})'
    else :
        return '请在RequestBody中加入您要选择训练的公司编号(如：{"code":"601998.SH"})'


@app.route('/predictWithPath', methods=["GET","POST"])
def predictWithPath():
    # data = {"success": False}
    # params = flask.request.json  
    # if (params != None):
    #     if (params['path']!= None):
    #         path = params['path']
    #         ts_code = path[19:25]+"."+path[25:27]
    #         step_length = 60
    #         now = datetime.datetime.now()
    #         delta = datetime.timedelta(days=730)
    #         n_days = now - delta
    #         start_date = n_days.strftime('%Y%m%d')
    #         end_date = now.strftime('%Y%m%d')


    #         df = pro.daily(start_date=start_date, end_date=end_date, ts_code=ts_code)
    #         close = df["close"].values.astype('float64')
    #         close = close[::-1]
    #         close = close.reshape(-1,1)
    #         scaler_close = MinMaxScaler(feature_range=(0,1))
    #         close = scaler_close.fit_transform(close)
            
    #         vol = df["vol"].values.astype('float64')
    #         vol = vol[::-1]
    #         vol = vol.reshape(-1,1)
    #         scaler_vol = MinMaxScaler(feature_range=(0,1))
    #         vol = scaler_vol.fit_transform(vol)
    #         dataset = np.concatenate((close,vol),axis=1)

    #         open_data = df["open"].values.astype('float64')
    #         open_data = open_data[::-1]
    #         open_data = open_data.reshape(-1,1)
    #         scaler_open = MinMaxScaler(feature_range=(0,1))
    #         open_data = scaler_open.fit_transform(open_data)
    #         dataset = np.concatenate((dataset,open_data),axis=1)

    #         high = df["high"].values.astype('float64')
    #         high = high[::-1]
    #         high = high.reshape(-1,1)
    #         scaler_high = MinMaxScaler(feature_range=(0,1))
    #         high = scaler_high.fit_transform(high)
    #         dataset = np.concatenate((dataset,high),axis=1)

    #         low = df["low"].values.astype('float64')
    #         low = low[::-1]
    #         low = low.reshape(-1,1)
    #         scaler_low = MinMaxScaler(feature_range=(0,1))
    #         low = scaler_low.fit_transform(low)
    #         dataset = np.concatenate((dataset,low),axis=1)


    #         test_size = 90
    #         train_size = len(dataset) - 90
    #         test_set = dataset[train_size:len(dataset),:]

    #         x_test = []
    #         y_test = []
    #         # 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
    #         for i in range(step_length, len(test_set)-1):
    #             x_test.append(test_set[i - step_length:i,:])
    #             y_test.append(test_set[i, 0])
    #         # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
    #         x_test, y_test = np.array(x_test), np.array(y_test)
    #         x_test = np.expand_dims(x_test, axis=2)

    #         model = tf.keras.models.load_model(path)
    #         #反归一化成原始数据，一会作比较
    #         real_test_set = test_set.copy()
    #         real_stock_price = scaler_close.inverse_transform(real_test_set[:,0].reshape(-1,1))     #因为判断涨跌正确率需要前一天数据，所以比预测多一天，下面会分别处理
    #         real_stock_price = real_stock_price[step_length-1:]
    #         real_stock_price_show=[]
    #         for i in real_stock_price:
    #             real_stock_price_show.append(i[0])

    #         #预测数据准备
    #         predicted_stock_price_final = []
    #         pre_set = []
            
    #         for i in range(step_length, len(test_set)):
    #             pre_set.append(test_set[i - step_length : i,:])
                
    #         test_set_use = test_set.copy()

    #         #预测
    #         count = 0
    #         for i in range(0,len(pre_set)):
    #             pre_set[i] = np.expand_dims(pre_set[i], axis=0)
    #             predicted_stock_price_tmp = model.predict(pre_set[i])
                
    #             predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
    #             test_set_use[i+step_length][0] = predicted_stock_price_tmp[0]   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

    #         predict_next_day = test_set_use[-1:,]
    #         test_set_use = test_set_use[step_length:-1,]
    #         predicted_stock_price_final = test_set_use[:,0]

    #         deviation = 0.0
    #         for i in range(0,len(real_stock_price_show)-2):
    #             deviation += abs(real_stock_price_show[i+1] - predicted_stock_price_final[i]) / real_stock_price_show[i+1]
    #         deviation /= (len(real_stock_price_show)-1)
    #         acc = 1 - deviation
    #         print("准确率：",acc*100,"%")

    #         acc_longOrShort = 0.0
    #         right_count = 0
    #         for i in range(0,len(real_stock_price_show)-2):
    #             if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predicted_stock_price_final[i] - real_stock_price_show[i]) >=0 :
    #                 right_count +=1
    #         acc_longOrShort = right_count / (len(real_stock_price_show)-2)
    #         print("涨跌正确率：",acc_longOrShort*100,"%")

    #         acc_profit = 1.0
    #         buy_count = 0
    #         for i in range(0,len(real_stock_price_show)-2):
    #             if predicted_stock_price_final[i] - real_stock_price_show[i]>=0 :
    #                 acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
    #                 buy_count+=1
    #         # else :

    #             #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

    #         acc_profit -= 1
    #         print("盈亏率：",acc_profit*100,"%")
        
    #         show = str(predict_next_day[0][0])
    #         data["该模型前29天准确率："] = str(acc*100)+"%"
    #         data["该模型前29天涨跌正确率："] = str(acc_longOrShort*100)+"%"
    #         data["该模型前29天买入次数："] = str(buy_count)+"次"
    #         data["该模型前29天盈亏率："] = str(acc_profit*100)+"%"
    #         data["明日收盘价预测值："] = show
    #         data["success"] = True
    step_length = 180
    data = {"success": False}
    params = flask.request.json  
    if (params != None):
        try:
            path = params['path']
        except:
            path = 1
        ts_code = path[19:25]+"."+path[25:27]
    #         step_length = 60
    #         now = datetime.datetime.now()
    #         delta = datetime.timedelta(days=730)
    #         n_days = now - delta
    #         start_date = n_days.strftime('%Y%m%d')
    #         end_date = now.strftime('%Y%m%d')

        now = datetime.datetime.now()
        delta = datetime.timedelta(days=2000)
        n_days = now - delta
        start_date = n_days.strftime('%Y%m%d')
        end_date = now.strftime('%Y%m%d')
       # df = pro.index_basic(market = 'SSE')
        df = pro.daily(start_date=start_date, end_date=end_date, ts_code=ts_code)
        print(df)
        close = df["close"].values.astype('float64')
        close = close[::-1]
        close = close.reshape(-1,1)
        scaler_close = MinMaxScaler(feature_range=(0,1))
        close = scaler_close.fit_transform(close)
        
        vol = df["vol"].values.astype('float64')
        vol = vol[::-1]
        vol = vol.reshape(-1,1)
        scaler_vol = MinMaxScaler(feature_range=(0,1))
        vol = scaler_vol.fit_transform(vol)
        dataset = np.concatenate((close,vol),axis=1)

        open_data = df["open"].values.astype('float64')
        open_data = open_data[::-1]
        open_data = open_data.reshape(-1,1)
        scaler_open = MinMaxScaler(feature_range=(0,1))
        open_data = scaler_open.fit_transform(open_data)
        dataset = np.concatenate((dataset,open_data),axis=1)

        high = df["high"].values.astype('float64')
        high = high[::-1]
        high = high.reshape(-1,1)
        scaler_high = MinMaxScaler(feature_range=(0,1))
        high = scaler_high.fit_transform(high)
        dataset = np.concatenate((dataset,high),axis=1)

        low = df["low"].values.astype('float64')
        low = low[::-1]
        low = low.reshape(-1,1)
        scaler_low = MinMaxScaler(feature_range=(0,1))
        low = scaler_low.fit_transform(low)
        dataset = np.concatenate((dataset,low),axis=1)



        test_size = 210
        train_size = len(dataset) - 210
        test_set = dataset[train_size:len(dataset),:]


        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_test_set = test_set.copy()
        real_stock_price = scaler_close.inverse_transform(real_test_set[:,0].reshape(-1,1))     #因为判断涨跌正确率需要前一天数据，所以比预测多一天，下面会分别处理
        real_stock_price = real_stock_price[step_length-1:]
        real_stock_price_show=[]
        for i in real_stock_price:
            real_stock_price_show.append(i[0])

        #处理预测数据
        predicted_stock_price_final = []
        pre_set = []
        for i in range(step_length, len(test_set)):
            pre_set.append(test_set[i - step_length : i,:])

        test_set_use = test_set.copy()
        model = tf.keras.models.load_model(path)
        days = int(path[-1])
        if (days == 1):
            #model = tf.saved_model.load("D:/model/300/180_1day/16_32_gru_sigmoid_huber_sdg_b32")
            #预测
            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])  #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_1day = predict_arr[-1,:]

            deviation = 0.0
            for i in range(0,len(real_stock_price_show)-2):
                deviation += abs(real_stock_price_show[i+1] - predict_arr[i]) / real_stock_price_show[i+1]
            deviation /= (len(real_stock_price_show)-1)
            acc = 1 - deviation
            print("准确率：",acc*100,"%")

            acc_longOrShort = 0.0
            right_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if (real_stock_price_show[i+1] - real_stock_price_show[i])*(predict_arr[i] - real_stock_price_show[i]) >=0 :
                    right_count +=1
            acc_longOrShort = right_count / (len(real_stock_price_show)-2)
            print("涨跌正确率：",acc_longOrShort*100,"%")

            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predict_arr[i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(buy_count)
            show = str(predict_next_1day)
            show = show[1:-1]
            print(show)
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"


        elif(days == 3):
            #预测
            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_3day = predict_arr[-1,:]
            #test_set_use = test_set_use[step_length:-1,].

            #predicted_stock_price = np.zeros((3),dtype= np.float64)
            predicted_stock_price_list = []
            for i in range(0,3):
                predicted_stock_price_list.append(predict_arr[:,i])
            predicted_stock_price = np.array(predicted_stock_price_list)


            deviation = np.zeros((3),dtype= np.float64)
            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,3):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                        deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

            acc = np.zeros((3),dtype= np.float64)
            
            for i in range(0,3):
                deviation[i] /= (len(real_stock_price_show)-1-i)
                acc[i] = 1 -deviation[i]

            #三天的涨跌准确率
            acc_longOrShort = np.zeros((3),dtype= np.float64)
            right_count = np.zeros((3),dtype= np.float64)

            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,3):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3) or j==0):
                        if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                            right_count[j] += 1

            for i in range(0,3):
                acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



            #三天的收益率
            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(predict_next_3day)
            # show = str(predict_next_3day)
            # print(show)
            show = ''.join(str(i)+"," for i in predict_next_3day)
            show = show[:-1]
            print(show)
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"

        elif(days == 5):
            #预测
            count = 0
            predict_list = []
            for i in range(0,len(pre_set)):
                pre_set[i] = np.expand_dims(pre_set[i], axis=0)
                predicted_stock_price_tmp = model(pre_set[i])
                predicted_stock_price_tmp = scaler_close.inverse_transform(predicted_stock_price_tmp)
                predict_list.append(predicted_stock_price_tmp[0])   #test_set_use前二十九个用来比较，最后一个用来当作预测结果  

            predict_arr = np.array(predict_list)
            predict_next_5day = predict_arr[-1,:]
            #test_set_use = test_set_use[step_length:-1,].

            #predicted_stock_price = np.zeros((3),dtype= np.float64)
            predicted_stock_price_list = []
            for i in range(0,days):
                predicted_stock_price_list.append(predict_arr[:,i])
            predicted_stock_price = np.array(predicted_stock_price_list)


            deviation = np.zeros((days),dtype= np.float64)
            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,days):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                        deviation[j] += abs(real_stock_price_show[i+1+j] - predicted_stock_price[j][i]) / real_stock_price_show[i+1]

            acc = np.zeros((days),dtype= np.float64)
            
            for i in range(0,days):
                deviation[i] /= (len(real_stock_price_show)-1-i)
                acc[i] = 1 -deviation[i]

            #五天的涨跌准确率
            acc_longOrShort = np.zeros((days),dtype= np.float64)
            right_count = np.zeros((days),dtype= np.float64)

            for i in range(0,len(real_stock_price_show)-1):
                for j in range(0,days):
                    if((j==1 and i<len(real_stock_price_show)-2)or(j==2 and i<len(real_stock_price_show)-3)or(j==3 and i<len(real_stock_price_show)-4)or(j==4 and i<len(real_stock_price_show)-5) or j==0):
                        if (real_stock_price_show[i+1+j] - real_stock_price_show[i])*(predicted_stock_price[j][i] - real_stock_price_show[i]) >=0 :
                            right_count[j] += 1

            for i in range(0,days):
                acc_longOrShort[i] = right_count[i] / (len(real_stock_price_show)-1-i)



            #五天的收益率
            acc_profit = 1.0
            buy_count = 0
            for i in range(0,len(real_stock_price_show)-2):
                if predicted_stock_price[0][i] - real_stock_price_show[i]>=0 :
                    acc_profit = acc_profit * ((real_stock_price_show[i+1] - real_stock_price_show[i]) / real_stock_price_show[i] + 1)
                    buy_count+=1
            # else :

                #     acc_profit = acc_profit * ((real_stock_price_show[i] - real_stock_price_show[i+1]) / real_stock_price_show[i] + 1)

            acc_profit -= 1
            print("盈亏率：",acc_profit*100,"%")
            print(buy_count)
            show = ''.join(str(i)+"," for i in predict_next_5day)
            show = show[:-1]
            data["acc"] = str(acc)
            data["predict"] = show
            data["acc_longOrShort"] = str(acc_longOrShort)
            data["该模型前29天买入次数"] = str(buy_count)+"次"
            data["该模型前29天盈亏率"] = str(acc_profit*100)+"%"
            
        trade_date = df["trade_date"].values.astype('int')
        trade_date = trade_date[::-1]
        trade_date = trade_date.reshape(-1,1)
        before_date = ''.join(str(i[0])+"," for i in trade_date[-10:])
        before_date = before_date[:-1]

        real_10days = ''.join(str(i)+"," for i in real_stock_price_show[-10:])
        real_10days = real_10days[:-1]

        data["before_date"] = before_date
        data["real_10days"] = real_10days
        data["success"] = True
            # 返回Json格式的响应
        return flask.jsonify(data)  
    #     else:
    #         return '请在body中加入您已有的训练模型路径(没有可调用/trainAndPredict接口训练属于自己的模型)'
    # else :
    #     return '请在body中加入您已有的训练模型路径(没有可调用/trainAndPredict接口训练属于自己的模型)'


# 启动Flask应用程序，允许远程连接
if __name__ == '__main__':
    app.run(debug=False,host=host)