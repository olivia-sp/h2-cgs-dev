import os
import io
import boto3
import json
import csv

import pandas as pd
import numpy as np

# Time 
import pytz
import datetime 

# weather
import urllib.request
import json

openweather_api_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
service_key = ",,,"

# Holiday Data    
import holidays


# # Model Load
import pickle
from pickle import load

import sklearn
from sklearn.preprocessing import MinMaxScaler


# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

# s3 download & lambda file save
BUCKET_NAME = 'sagemaker-project-,,,'
OBJECT_NAME = 'scaler/minmax_scaler.pkl'
FILE_NAME = '/tmp/minmax_scaler.pkl'

s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, OBJECT_NAME, FILE_NAME)

STATION_AXIS_INFO = {
                'congress'     :  {'nx': 59, 'ny': 126},
                # 'hanam'        :  {'nx' : 64, 'ny' : 126},
                # 'seongju'      :  {'nx' : 83, 'ny' : 93},
                # 'deokdong'     :  {'nx' : 89, 'ny' : 76},
                # 'palryong'     :  {'nx' : 90, 'ny' : 77},
                # 'anseong_sang' :  {'nx' : 63, 'ny' : 115},
                # 'sangam'       :  {'nx' : 58, 'ny' : 127},
                # 'yeoju'        :  {'nx' : 71, 'ny' : 121},
}


def ConcatDataCol(data1, data2):
    # 데이터 병합
    dataset = pd.concat([data1, data2], axis = 1) # col
    return dataset
    
def LoadTime_DF():
    UTC = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(UTC) # now.year, now.month, now.day, now.hour

    pd_now = pd.DataFrame({'year':[now.year], 
                            'month':[now.month], 
                            'day':[now.day], 
                            'hour':[now.hour]})

    return pd_now

def LoadTime():
    UTC = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(UTC) # now.year, now.month, now.day, now.hour
    now.strftime('%Y-%m-%d %H:%M:%S')

    return now


def DayOfWeek(df):
    # UTC = pytz.timezone('Asia/Seoul')
    # week = datetime.datetime.now(UTC).weekday()
    week_df = pd.DataFrame(columns=['fri','mon','sat','sun','thu','tue','wen','weekend','weekday'])
    
    for index in range(0,len(df)):
        week = datetime.date(df.loc[index,'year'], df.loc[index,'month'], df.loc[index,'day']).weekday()
        #print('week : ', week)

        if week is 0 :
            #week_df = pd.DataFrame({'fri':0,'mon':1,'sat':0,'sun':0,'thu':0,'tue':0,'wen':0,'weekend':0,'weekday':1}, index=[0])
            week_df.loc[index]=[0, 1, 0, 0, 0, 0, 0, 0, 1]
        elif week is 1:
            #week_df = pd.DataFrame({'fri':0,'mon':0,'sat':0,'sun':0,'thu':0,'tue':1,'wen':0,'weekend':0,'weekday':1}, index=[0])
            week_df.loc[index]=[0, 0, 0, 0, 0, 1, 0, 0, 1]
        elif week is 2:
            #week_df = pd.DataFrame({'fri':0,'mon':0,'sat':0,'sun':0,'thu':0,'tue':0,'wen':1,'weekend':0,'weekday':1}, index=[0])
            week_df.loc[index]=[0, 0, 0, 0, 0, 0, 1, 0, 1]
        elif week is 3:
            #week_df = pd.DataFrame({'fri':0,'mon':0,'sat':0,'sun':0,'thu':1,'tue':0,'wen':0,'weekend':0,'weekday':1}, index=[0])
            week_df.loc[index]=[0, 0, 0, 0, 1, 0, 0, 0, 1]
        elif week is 4:
            #week_df = pd.DataFrame({'fri':1,'mon':0,'sat':0,'sun':0,'thu':0,'tue':0,'wen':0,'weekend':0,'weekday':1}, index=[0])
            week_df.loc[index]=[1, 0, 0, 0, 0, 0, 0, 0, 1]
        elif week is 5:
            #week_df = pd.DataFrame({'fri':0,'mon':0,'sat':1,'sun':0,'thu':0,'tue':0,'wen':0,'weekend':0,'weekday':0}, index=[0])
            week_df.loc[index]=[0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif week is 6:
            #week_df = pd.DataFrame({'fri':0,'mon':0,'sat':0,'sun':1,'thu':0,'tue':0,'wen':0,'weekend':0,'weekday':0}, index=[0])
            week_df.loc[index]=[0, 0, 0, 1, 0, 0, 0, 0, 0]
        #print(week_df)

    return week_df


def getNowCity(STATION_NX_PAR, STATION__NY_PAR) :
    global openweather_api_url, service_key
    
    # to setup request URL
    pageNo = '1'
    numOfRows = '289' #24h
    dataType = 'JSON'

    now = LoadTime()
    now_date = now.strftime('%Y%m%d')
    now_hour = now.strftime('%H')
    print("now_date : ", now_date)
    base_date = str(now_date)
    base_time = '0500'
    
    '''
    단기예보
    - Base_time : 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300 (1일 8회)
    - API 제공 시간(~이후) : 02:10, 05:10, 08:10, 11:10, 14:10, 17:10, 20:10, 23:10
    
    - 강수형태(PTY) 코드 : (단기) 없음(0), 비(1), 비/눈(2), 눈(3), 소나기(4) - clear(0)/rain(1-4)으로 통합
    '''

    nx = str(STATION_NX_PAR)
    ny = str(STATION__NY_PAR)
    
    PARAMS = {
            'serviceKey' : service_key,
            'pageNo' : pageNo,
            'numOfRows' : numOfRows,
            'dataType'  : dataType,
            'base_date' : base_date,
            'base_time' : base_time,
            'nx' : nx,
            'ny' : ny 
        }
    
    # API 요청시 필요한 인수값 정의
    ow_api_url = openweather_api_url
    payload = "?serviceKey=" + service_key + "&pageNo=" + pageNo + "&numOfRows=" + numOfRows + "&dataType=" + dataType + "&base_date=" + base_date + "&base_time=" + base_time + "&nx=" + nx + "&ny=" + ny
    url_total = ow_api_url + payload
    #print(url_total)
    

    # API 요청하여 데이터 받기
    req = urllib.request.urlopen(url_total)
    data = json.loads(req.readline())
    #print (data)
    dataArray = data['response']['body']['items']['item']
    
    pd_weather = pd.DataFrame(columns=['year', 'month', 'day', 'hour', 'temp', 'humid', 'precipitation', 'clear', 'rain'])
    weather_date = []
    weather_year = []
    weather_month = []
    weather_day = []
    weather_hour = []
    weather_clear = []
    weather_rain = []
    weather_preci = []
    weather_temp = []
    weather_humid = []
    weather_fcst_data = []
    
    for idx in range(0,len(dataArray)):
        if dataArray[idx]['category'] == 'PTY': # 강수 형태
            v = dataArray[idx]['fcstValue']
            if v == 0:
                weather_clear.append(1)
                weather_rain.append(0)
            elif v != 0:
                weather_clear.append(0)
                weather_rain.append(1)
        elif dataArray[idx]['category'] == 'REH': # 습도
            weather_humid.append(dataArray[idx]['fcstValue'])
        elif dataArray[idx]['category'] == 'PCP': # 1시간 강수량
            if dataArray[idx]['fcstValue'] == '강수없음':
                weather_preci.append(0)
            else:
                v = dataArray[idx]['fcstValue']
                if v.find("mm") != -1:
                    print("Found string")
                    new_str = v.replace('mm', '')
                    weather_preci.append(new_str)
        elif dataArray[idx]['category'] == 'TMP': # 기온
            weather_temp.append(dataArray[idx]['fcstValue'])
            weather_date.append(dataArray[idx]['fcstDate'])
            #print(int(dataArray[idx]['fcstDate'])/10000)
            
            d_year = int(int(dataArray[idx]['fcstDate'])/10000)
            d_month = int((int(dataArray[idx]['fcstDate'])%10000)/100)
            d_day = int((int(dataArray[idx]['fcstDate'])%10000)%100)
            d_hour = int(dataArray[idx]['fcstTime'])/100
            
            weather_year.append(d_year)
            weather_month.append(d_month)
            weather_day.append(d_day)
            weather_hour.append(d_hour)
            
        else:
            pass
    weather_info = {
            'date'          : weather_date,
            'year'          : weather_year,
            'month'         : weather_month,
            'day'           : weather_day,
            'hour'          : weather_hour,
            'temperature'   : weather_temp,
            'humid'         : weather_humid,
            'precipitation' : weather_preci,
            'clear'         : weather_clear,
            'rain'          : weather_rain,
        }
        
    pd_weather = pd.DataFrame(weather_info)
    #print("pd_weather:", pd_weather)
        
    return pd_weather

def DefineHoliday(df):
    day_list = df['date']
    
    # 한국 휴일 개체 생성
    kr_holidays = holidays.KR()

    holiday_df = pd.DataFrame(columns=['date', 'holiday', 'non_holiday'])
    holiday_df['date'] = day_list
    holiday_df['holiday'] = holiday_df.date.apply(lambda x: 1 if x in kr_holidays else 0)
    holiday_df['non_holiday'] = holiday_df.date.apply(lambda x: 0 if x in kr_holidays else 1)

    #print("holiday_df : ", holiday_df)    

    return holiday_df


def lambda_handler(event, context):
    global STATION_AXIS_INFO
        
    response_status = 200
    response_body = []

    station_name = 'congress'

    nx = str(STATION_AXIS_INFO[station_name]['nx'])
    ny = str(STATION_AXIS_INFO[station_name]['ny'])    
    print(nx, ny)
    
    # Make Input DataFrame 
    csv_new_df_dataset = pd.DataFrame()
    now = LoadTime()
    pd_now = LoadTime_DF()
    pd_weather = getNowCity(nx, ny)
    holiday_df = DefineHoliday(pd_weather)
    week_df = DayOfWeek(pd_weather)
    df_dataset = ConcatDataCol(pd_weather, holiday_df)
    df_dataset = ConcatDataCol(df_dataset, week_df)
    df_dataset = df_dataset.drop(['date'], axis = 1)
    #print("input_dataset_columns;", df_dataset.columns)
    # print("input_dataset:", df_dataset)
    
    # Scaler Model Load
    print(os.listdir("/tmp"))
    loaded_minmax_scaler = load(open(FILE_NAME, 'rb'))
    scaled_x = pd.DataFrame(loaded_minmax_scaler.transform(df_dataset))
    #print("input_dataset_scaled:", scaled_x)

    
    # Fommating input data : json
    for index in range(0,len(scaled_x)):
        result = scaled_x.loc[index].to_json(orient="values")
        # print("result : ", result)
        result = result.replace('[', '')
        result = result.replace(']', '')
        
        # print("input_dataset_json:", result)
        event = {
            'data' : result
        }

        # Sagemaker Endpoint Invoke
        try:
            #print("Received event: " + json.dumps(event, indent=2))
            data = json.loads(json.dumps(event))
            payload = data['data']
            #print("payload : ", payload)
            
            
            response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                              ContentType='text/csv',
                                              Body=payload)
            #print("response : ", response)
            result = json.loads(response['Body'].read().decode())
            result_info = {"total_car_num":result}
            response_body.append(result_info)
            

        except Exception as e:
            response_status = 500
            # response_body = "Server Error"
            response_body = json.dumps(e)
            pass
    
    print("response_body : ", response_body)
            
    
    
    response = {
        'statusCode': response_status,
        'body': json.dumps(response_body)
    }
    return response
