# -*- coding: utf-8 -*-
# @Time : 2022/11/26 16:23 
# @Author : Mingzheng 
# @File : data_processing.py
# @desc :
import random
import shutil
import itertools
import moviepy
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from datetime import datetime
from moviepy.editor import *
from moviepy.audio.io.AudioFileClip import AudioFileClip
from utils.utils import mkdir_if_missing,video_clip
import librosa
import librosa.display
from pydub import AudioSegment
import wave
from ast import literal_eval
import soundfile as sf


def min_max_normalization(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x-x_min)/(x_max-x_min)
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

class utils_audio():
    def __init__(self):
        self.dataset_path_train = r'D:\CMZ\dataset_new\audio_without_background_split_nocoverage_train'
        self.dataset_path_test = r'D:\CMZ\dataset_new\audio_without_background_split_nocoverage_test'
        self.dataset_path = r'D:\CMZ\dataset\audio_split\with_background\audio'
        self.n_mfcc = 40
    def get_distance(self,distances):
        # 如果最后一个噪音的结束时间戳到音频duration之间相差不到三秒，就放弃
        if (distances[1]-distances[0])<3:return True
        elif (distances[1]-distances[0])>3:return False
    def splited_with_background(self,audio_path,save_path,background_path):
        '''
        根据噪音信息切割音频
        :param audio_path:
        :param save_path:
        :param background_path:
        :return:
        '''
        audio = AudioFileClip(audio_path)
        duration = int(audio.duration)      # 音频总时长，s
        begin = 0
        end = duration
        audio_name = os.path.split(audio_path)[1][:-4]
        background_information = pd.read_csv(background_path)
        background_information = background_information[background_information['filename']==audio_name]
        if background_information.__len__()==0:
            shutil.copyfile(audio_path,os.path.join(save_path,os.path.split(audio_path)[1]))
        else:
            background_information_start_end = background_information['n_start'].astype(str).str.cat(background_information['n_end'].astype(str),sep=',').tolist()#将前后时间戳字符串组合
            background_information_start_end = list(map(eval,background_information_start_end))     #将前后时间戳字符串转为列表
            cut_frames = []     #切割时间戳集合
            for i in range(len(background_information_start_end)):
                if i == 0:
                    cut_frame = [begin,background_information_start_end[i][0]]
                    if not self.get_distance(cut_frame):cut_frames.append(cut_frame)
                elif i == len(background_information_start_end)-1:
                    cut_frame1 = [background_information_start_end[i-1][1],background_information_start_end[i][0]]
                    cut_frame2 = [background_information_start_end[i][1],duration]
                    if not self.get_distance(cut_frame1): cut_frames.append(cut_frame1)
                    if not self.get_distance(cut_frame2): cut_frames.append(cut_frame2)        #如果最后一个噪音的结束时间戳到音频duration之间相差不到三秒，就放弃
                else:
                    cut_frame = [background_information_start_end[i-1][1],background_information_start_end[i][0]]
                    if not self.get_distance(cut_frame): cut_frames.append(cut_frame)
            if len(cut_frames)==0:shutil.copyfile(audio_path,os.path.join(save_path,os.path.split(audio_path)[1]))
            for item in cut_frames:
                audio_cut = audio.subclip(item[0],item[1])
                audio_cut_name = audio_name+'_cut[{},{}]'.format(str(item[0]),str(item[1]))
                if not os.path.exists(os.path.join(save_path,audio_cut_name+'.mp3')):
                    audio_cut.write_audiofile(os.path.join(save_path,audio_cut_name+'.mp3'))
                else:
                    print('已存在跳过')
                    continue
    def audio_split(self,audio_path,fragment_length,overlap_length,save_path):
        '''
        音频切割
        :param audio_path: 音频路径
        :param fragment_length: 切割片段总长
        :param overlap_length: 切割片段重叠长度
        :param save_path: 保存路径
        :return:
        '''
        audio = AudioFileClip(audio_path)
        duration = int(audio.duration)    #音频总时长，s
        begin,end = 0,0
        audio_name = os.path.split(audio_path)[1][:-4]
        #日志变量
        num = 1
        while True:
            if end > duration or begin > duration-fragment_length: break
            end = min(begin+fragment_length,duration)
            audio_cut = audio.subclip(begin,end)
            audio_cut_save_name = audio_name+'_segment[{},{}]'.format(str(begin),str(end))
            audio_cut_save_path = os.path.join(save_path,audio_cut_save_name+'.mp3')
            audio_cut.write_audiofile(audio_cut_save_path)
            begin += overlap_length
            print('已切割：{}份'.format(str(num)))
            num+=1

    def audio_argument(self, audio_path, save_path):
        '''
        音频切割
        :param audio_path: 音频路径
        :param fragment_length: 切割片段总长
        :param overlap_length: 切割片段重叠长度
        :param save_path: 保存路径
        :return:
        '''

        files = glob.glob(os.path.join(audio_path, '*.mp3'))
        labels = list(map(lambda x: os.path.split(x)[-1].split('_'), files))
        Y = pd.DataFrame(labels).iloc[:, 1:6]  # 提取雨量，温度，湿度，气压，风速，
        Y.columns = ['RAINFALL INTENSITY', 'TEMPORTURE', 'HUMIDITY', 'ATMOSPHERE PHERE', 'WIND SPEED']
        files_needed_index = Y[Y['RAINFALL INTENSITY'].astype(float)>10].index.tolist()
        # timestretch扩增
        rates = [0.81, 1.07]
        for item in tqdm(files_needed_index):
            file_path = files[item]
            # timestretch扩增
            # for rate in rates:
            #     save_name = os.path.split(file_path)[1]+'_argument(timestretch{})'.format(str(rate))+'.mp3'
            #     file_save_path = os.path.join(save_path,save_name)
            #     y, sr = librosa.load(file_path)
            #     y_changed = librosa.effects.time_stretch(y, rate=rate)
            #     sf.write(file_save_path,y_changed,sr)
            # pitch shifting扩增
            tone_steps = [-1, -2, 1, 2]
            for tone_step in tone_steps:
                save_name = os.path.split(file_path)[1][:-4] + '_argument(pitch shifting{})'.format(str(tone_step)) + '.mp3'
                file_save_path = os.path.join(save_path, save_name)
                y, sr = librosa.load(file_path)
                y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
                sf.write(file_save_path, y_changed, sr)

    def eliminate_noise(self,audio_path,noise_list,save_path):
        '''
        根据噪音标注去除噪音
        :param audio_path:
        :param noise_list: 噪音信息列表
        :param save_path:
        :return:
        '''
        audio_time = self.get_wav_time(audio_path)
        audio_name = os.path.split(audio_path)[1].split('.')[0]
        # 如果噪音列表不为空,则去除
        if noise_list != None:
            for index,item in enumerate(noise_list):
                if index == 0 :
                    audio_cut_file = self.audio_cut(audio_path,0,item[0])
                    audio_save_path = os.path.join(save_path,audio_name+'(0,'+str(item[0])+').wav')
                elif index == len(noise_list) - 1:
                    audio_cut_file = self.audio_cut(audio_path,item[1],audio_time)
                    audio_save_path = os.path.join(save_path, audio_name + '('+str(item[1])+',' + str(audio_time) + ').wav')
                else:
                    if item[1] == noise_list[index+1][0]:continue
                    audio_cut_file = self.audio_cut(audio_path,item[1],noise_list[index+1][0])
                    audio_save_path = os.path.join(save_path, audio_name + '('+str(item[1])+',' + str(noise_list[index+1][0]) + ').wav')
                audio_cut_file.export(audio_save_path,format="wav")
                print(audio_save_path)
        else:
            audio_save_path = os.path.join(save_path,os.path.split(audio_path)[1])
            shutil.copy(audio_path,audio_save_path)
    def get_mfcc(self,file_path, mfcc_max_padding=0, n_mfcc=400):
        try:
            # Load audio file
            y, sr = librosa.load(file_path)

            # Normalize audio data between -1 and 1
            normalized_y = librosa.util.normalize(y)

            # Compute MFCC coefficients
            mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=n_mfcc)

            # Normalize MFCC between -1 and 1
            normalized_mfcc = librosa.util.normalize(mfcc)

            # Should we require padding
            shape = normalized_mfcc.shape[1]
            if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff // 2
                xRight = xDiff - xLeft
                normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0, 0), (xLeft, xRight)), mode='constant')

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None
        return normalized_mfcc
    def get_mfcc_geo(self,file_path,geo, mfcc_max_padding=0, n_mfcc=400):
        try:
            # Load audio file
            y, sr = librosa.load(file_path)

            # Normalize audio data between -1 and 1
            normalized_y = librosa.util.normalize(y)

            # Compute MFCC coefficients
            mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=n_mfcc)*geo
            mfcc = mfcc*geo
            # Normalize MFCC between -1 and 1
            normalized_mfcc = librosa.util.normalize(mfcc)

            # Should we require padding
            shape = normalized_mfcc.shape[1]
            if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff // 2
                xRight = xDiff - xLeft
                normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0, 0), (xLeft, xRight)), mode='constant')

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None
        return normalized_mfcc

    def get_mel_spectrogram(self,file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
        try:
            # Load audio file
            y, sr = librosa.load(file_path)

            # Normalize audio data between -1 and 1
            normalized_y = librosa.util.normalize(y)

            # Generate mel scaled filterbanks
            mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

            # Convert sound intensity to log amplitude:
            mel_db = librosa.amplitude_to_db(abs(mel))

            # Normalize between -1 and 1
            normalized_mel = librosa.util.normalize(mel_db)

            # Should we require padding
            shape = normalized_mel.shape[1]
            if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff // 2
                xRight = xDiff - xLeft
                normalized_mel = np.pad(normalized_mel, pad_width=((0, 0), (xLeft, xRight)), mode='constant')

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None
        return normalized_mel
    def get_stft(self,file_path, mfcc_max_padding=0):
        try:
            # Load audio file
            y, sr = librosa.load(file_path)

            # Normalize audio data between -1 and 1
            normalized_y = librosa.util.normalize(y)
            # Windowing
            n_fft = 2048
            hop_length = 512
            # Generate mel scaled filterbanks
            stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)

            # Convert sound intensity to log amplitude:
            stft_db = librosa.amplitude_to_db(abs(stft))

            # Normalize between -1 and 1
            normalized_stft = librosa.util.normalize(stft_db)

            # Should we require padding
            shape = normalized_stft.shape[1]
            if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff // 2
                xRight = xDiff - xLeft
                normalized_mel = np.pad(normalized_stft, pad_width=((0, 0), (xLeft, xRight)), mode='constant')

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None
        return normalized_stft
    def add_padding(self,features, mfcc_max_padding=174):
        padded = []

        # Add padding
        for i in range(len(features)):
            px = features[i]
            size = len(px[0])
            # Add padding if required
            if (size < mfcc_max_padding):
                xDiff = mfcc_max_padding - size
                xLeft = xDiff // 2
                xRight = xDiff - xLeft
                px = np.pad(px, pad_width=((0, 0), (xLeft, xRight)), mode='constant')

            padded.append(px)

        return padded
    def preprocessing(self,file_path,):
        files = glob.glob(os.path.join(file_path,'*.mp3'))
        labels = list(map(lambda x:os.path.split(x)[-1].split('_'),files))
        Y = pd.DataFrame(labels).iloc[:,1:6]   #提取雨量，温度，湿度，气压，风速，
        Y.columns=['RAINFALL INTENSITY','TEMPORTURE','HUMIDITY','ATMOSPHERE PHERE','WIND SPEED']

        features = []
        frames_max = 0
        for index,file_path in enumerate(tqdm(files)):
            # Extract MFCCs (do not add padding)
            mfccs = self.get_mfcc(file_path,0,self.n_mfcc)
            # Save current frame count
            num_frames = mfccs.shape[1]
            # Add row (feature)
            features.append(mfccs)
            # Update frames maximum
            if (num_frames > frames_max):
                frames_max = num_frames
        padded = []
        # Add padding
        mfcc_max_padding = frames_max
        for i in range(len(features)):
            size = len(features[i][0])
            if (size < mfcc_max_padding):
                pad_width = mfcc_max_padding - size
                px = np.pad(features[i],
                            pad_width=((0, 0), (0, pad_width)),
                            mode='constant',
                            constant_values=(0,))
            else:
                px = features[i]
            padded.append(px)
        X = np.array(padded)
        return X,Y
    def rainfall_classify(self,x):
        if x['RAINFALL INTENSITY']<=0.4:return 'light'
        elif x['RAINFALL INTENSITY']>0.4 and x['RAINFALL INTENSITY']<=1.24: return 'moderate'
        elif x['RAINFALL INTENSITY']>=1.24 and x['RAINFALL INTENSITY'] <= 2.49: return 'heavy'
        elif x['RAINFALL INTENSITY']>2.49:return 'violent'
        else: return 'none'

class utils_file():
    def __init__(self):
        pass
    def rainfall_classify(self,x):
        if x['RAINFALL INTENSITY']<=0.4:return 'light'
        elif x['RAINFALL INTENSITY']>0.4 and x['RAINFALL INTENSITY']<=1.24: return 'moderate'
        elif x['RAINFALL INTENSITY']>=1.24 and x['RAINFALL INTENSITY'] <= 2.49: return 'heavy'
        elif x['RAINFALL INTENSITY']>2.49:return 'violent'
        else: return 'none'
    def convert_from_surveillance_camera(self,file_path,save_path,duration):
        '''
        将从相机内存卡内拿到的视频转为正常视频文件
        :param file_path:
        :param save_path:
        :param duration: 视频时长
        :return:
        '''
        start_time_stamp = (0,0,0)
        video_clip(file_path,save_path,start_time_stamp,duration)

    def infer_time_stamp(self,target_time,video_start_time,video_duration):
        '''
        推断视频截取的起始时间戳,结尾时间戳
        :param target_time: 目标时间戳
        :param video_start_time:视频起始时间戳
        :param video_duration:视频时长
        :return:
        '''
        t_hour = target_time.hour
        t_minute = target_time.minute
        t_second = target_time.second
        v_hour = video_start_time.hour
        v_minute = video_start_time.minute
        v_second = video_start_time.second
        t_hour_back = t_hour
        t_minute_back = t_minute - 1
        t_second_back = t_second
        t_time_stamp_back = str(target_time).split(' ')[0] + ' ' + str(t_hour_back) + ':' + str(t_minute_back) + ':' + str(t_second_back)
        if t_minute_back==v_minute:
            if t_second_back<v_second:t_time_stamp_back = str(target_time).split(' ')[0] + ' ' + str(t_hour_back)+':'+str(t_minute_back)+':'+str(v_second)
        difference_tback_vstart = datetime.strptime(t_time_stamp_back,'%Y-%m-%d %H:%M:%S')-video_start_time    #目标时间戳-1分钟与视频起始时间戳的距离
        c_start_time_stamp = difference_tback_vstart.seconds
        difference_t_vstart = target_time-video_start_time
        c_end_time_stamp = min(video_duration,difference_t_vstart.seconds)
        return [c_start_time_stamp,c_end_time_stamp]

    def generate_dataset(self,meteorological_path,video_file_time_path,video_path,video_save_path,audio_save_path):
        '''
        根据气象信息生成数据集
        :param methodological_path:气象数据
        video_file_time_path:视频名字，时间对应表格
        video_path:视频路径
        save_path:数据集保存路径
        :return:
        '''
        meteorological_information = pd.read_csv(meteorological_path)
        file_time_information = pd.read_csv(video_file_time_path)
        # 提取时间，构造day列，将格式转为%Y-%m-%d
        file_time_information['day'] = file_time_information['start'].map(lambda x:x.split(' ')[0])
        file_time_information['day'] = file_time_information['day'].map(lambda x:datetime.strptime(x,'%Y/%m/%d'))
        format_pattern = '%Y-%m-%d %H:%M:%S'
        # 筛选出雨量不是0的数据
        meteorological_information = meteorological_information[(meteorological_information['Rain_mm/h_1']!='0')&(meteorological_information['Rain_mm/h_1']!='mm/h')&(meteorological_information['Rain_mm/h_1']!='Smp')]
        # 获取各个变量
        for index in range(meteorological_information.shape[0]):
            item = meteorological_information.iloc[index]
            time_stamp = item['TIMESTAMP'].split(' ')
            date = time_stamp[0]
            time = time_stamp[1]
            temperature = round((float(item['Ta_up_Avg']) + float(item['Ta_low_Avg']))/2,3)    #上层温度加下层温度，取平均值
            humidity = round((float(item['RH_up_Avg']) + float(item['RH_low_Avg']))/2,3)      #上层湿度加下层湿度，取平均值
            atmosphere_pressure = round((float(item['e_up_Avg']) + float(item['e_low_Avg']))/2,3)      #上层气压加下层气压，取平均值
            wind_speed = item['ws_mean']
            rainfall_intensity = item['Rain_mm/h_1']
            # 与“文件-时间”的表格对比，截取视频
            file_time_information_sub = file_time_information[file_time_information['day']==date]
            for file_time_index in range(file_time_information_sub.shape[0]):
                # 通过时间差对比判断是否截取shipin
                file_time_information_sub_item = file_time_information_sub.iloc[file_time_index]
                file_time_information_sub_item_videofilename = file_time_information_sub_item['filename']
                file_time_information_sub_item_start = file_time_information_sub_item['start']
                file_time_information_sub_item_duration = datetime.strptime(file_time_information_sub_item['duration'],'%H:%M:%S')
                file_time_information_sub_item_duration_hour = file_time_information_sub_item_duration.hour
                file_time_information_sub_item_duration_minute = file_time_information_sub_item_duration.minute
                file_time_information_sub_item_duration_second = file_time_information_sub_item_duration.second
                file_time_information_sub_item_duration = file_time_information_sub_item_duration_hour*3600+file_time_information_sub_item_duration_minute*60+file_time_information_sub_item_duration_second
                target_time = datetime.strptime(item['TIMESTAMP'],'%Y-%m-%d %H:%M:%S')
                video_time = datetime.strptime(file_time_information_sub_item_start,'%Y/%m/%d %H:%M:%S')
                time_difference = video_time - target_time
                if time_difference.days==-1:time_difference = target_time-video_time
                # 若相隔时间小于视频时长则可以进行截取
                if time_difference.seconds<file_time_information_sub_item_duration:
                    # 若分钟数小于等于视频起始时间跳过
                    if target_time.hour < video_time.hour: continue
                    if target_time.minute<=video_time.minute:continue
                    try:
                        clip_start_end = self.infer_time_stamp(target_time,video_time,file_time_information_sub_item_duration)
                        clip_start = clip_start_end[0]
                        clip_end = clip_start_end[1]
                        clip_duration = clip_end-clip_start
                        video_file_path = os.path.join(video_path,file_time_information_sub_item_videofilename)
                        video_clip = VideoFileClip(video_file_path).subclip(clip_start, clip_end)
                        audio_clip = video_clip.audio
                        video_clip_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}.mp4'.format(str(target_time).replace(':','-'),str(rainfall_intensity),
                                                                            str(temperature),str(humidity),str(atmosphere_pressure),
                                                                            str(wind_speed),file_time_information_sub_item_videofilename[:-4],
                                                                                  str(clip_duration),file_time_information_sub_item['scenery'])
                        audio_clip_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}.mp3'.format(str(target_time).replace(':','-'), str(rainfall_intensity),
                                                                            str(temperature), str(humidity),
                                                                            str(atmosphere_pressure),
                                                                            str(wind_speed),
                                                                            file_time_information_sub_item_videofilename[:-4],
                                                                                  str(clip_duration),file_time_information_sub_item['scenery'])
                        video_clip_save_path = os.path.join(video_save_path,video_clip_name)
                        audio_clip_save_path = os.path.join(audio_save_path,audio_clip_name)
                        video_clip.write_videofile(video_clip_save_path)
                        audio_clip.write_audiofile(audio_clip_save_path)
                        with open('log.txt','a') as f:
                            f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(str(target_time),str(rainfall_intensity),
                                                                            str(temperature),str(humidity),str(atmosphere_pressure),
                                                                            str(wind_speed),file_time_information_sub_item_videofilename,clip_start,clip_end
                                                                            ,file_time_information_sub_item['scenery']))
                        print('processing {},{}'.format(file_time_information_sub_item_videofilename,str(target_time)))
                    except Exception as e:
                        with open('log_err.txt', 'a') as f:
                            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(str(target_time), str(rainfall_intensity),
                                                                          str(temperature), str(humidity),
                                                                          str(atmosphere_pressure),
                                                                          str(wind_speed),
                                                                          file_time_information_sub_item_videofilename,
                                                                          clip_start, clip_end,e))

    def spliy_train_test(self,file_path,train_path,test_path,file_back):
        '''
        分割训练测试
        :param file_path:
        :param train_path:
        :param test_path:
        :param file_back:文件后缀
        :return:
        '''
        files = glob.glob(os.path.join(file_path,'*.'+file_back))
        random.shuffle(files)
        split = int(0.7*len(files))
        files_train = files[0:split]
        files_test = files[split:]
        for item_train in tqdm(files_train):
            shutil.copy(item_train, os.path.join(train_path, os.path.split(item_train)[-1]))
        for item_test in tqdm(files_test):
            shutil.copy(item_test, os.path.join(test_path, os.path.split(item_test)[-1]))

class utils_static():
    def __init__(self):
        self.rainfall_intensity = {'small':[0,5/12],'middle':[5/12,14.9/12],'heavy':[15/12,29.9/12],'violent':[30/12,10000000]}


    def content_reading(self,path,back):
        if back == 'csv':return pd.read_csv(path)
        if back == 'xlsx':return pd.read_excel(path)

    def dataset_analysis(self,path,back):
        content = self.content_reading(path,back)
        rainfall = content['RAINFALL INTENSITY(mm/h)']
        scenery = content['SCENERY']
        scenery_num_dict = {}
        for scenery_name in scenery.unique():
            scenery_num_dict[scenery_name] = len([item for item in scenery if item == scenery_name])
        rainfall_intensity_num_dict = {}
        for rainfall_intensity_item in self.rainfall_intensity.keys():
            rainfall_intensity_num_dict[rainfall_intensity_item] = len([item for item in rainfall if item>= self.rainfall_intensity[rainfall_intensity_item][0] and item<= self.rainfall_intensity[rainfall_intensity_item][1]])
        print('test')
    def rainfall_intensity_static(self,path):
        '''
        统计分析雨量分布
        :param path:
        :return:
        '''
        audio_paths = glob.glob(os.path.join(path,'*.mp3'))
        labels = list(map(lambda x:os.path.split(x)[-1].split('_'),audio_paths))
        Y = pd.DataFrame(labels).iloc[:,1:6]   #提取雨量，温度，湿度，气压，风速，
        Y.columns=['RAINFALL INTENSITY','TEMPORTURE','HUMIDITY','ATMOSPHERE PHERE','WIND SPEED']
        Y = Y.astype('float32')
        Y['FILE NAME'] = audio_paths
        intensity_max = int(Y['RAINFALL INTENSITY'].max())+1
        intensity_min = 0
        intensity_grade = [item for item in range(intensity_min,intensity_max)]
        intensity_grade_combinations = {}
        distribution_result = {}
        for item in intensity_grade:
            intensity_grade_combinations[str(item)] = [item,item+1]
        for intensity_grade_key in intensity_grade_combinations.keys():
            intensity_grade_combination = intensity_grade_combinations[intensity_grade_key]
            left = intensity_grade_combination[0]
            right = intensity_grade_combination[1]
            distribution_result[intensity_grade_key] = len(Y[(Y['RAINFALL INTENSITY']>=left) & (Y['RAINFALL INTENSITY']<right)])
        return distribution_result

if __name__ == '__main__':

    file_utils = utils_file()
    audio_utils = utils_audio()
    static_utils = utils_static()

    '''
    # surveillance data ----->  normal video data
    
    # base_path = r'D:\CMZ\数据'
    # metadata_path = r'D:\CMZ\数据\file_name_time.csv'
    # metadata = pd.read_csv(metadata_path)
    # files_path = glob.glob(os.path.join(base_path,'*.mp4'))
    # convert_path = r'D:\CMZ\convert'
    # converted_files = glob.glob(os.path.join(convert_path,'*.mp4'))
    # for file_path in tqdm(files_path):
    #     file_name = os.path.split(file_path)[-1]
    #     if os.path.join(convert_path,file_name) in converted_files:continue
    #     if file_name == 'hiv00072.mp4':continue
    #     # if file_name == 'hiv00073.mp4':continue
    #     mkdir_if_missing(convert_path)
    #     duration = metadata[metadata['filename']==file_name]['duration'].values[0].split(':')
    #     hour = int(duration[0])
    #     minute = int(duration[1])
    #     seconds = int(duration[2])
    #     save_path = os.path.join(convert_path,file_name)
    #     file_utils.convert_from_surveillance_camera(file_path,save_path,(hour,minute,seconds))
    '''

    '''
    # static_utils = utils_static()
    # path = r'D:\CMZ\dataset\static_.xlsx'
    # static_utils.dataset_analysis(path,'xlsx')
    # split train and test

    # file_path = r'D:\CMZ\dataset\audio'
    # train_path = r'D:\CMZ\dataset\audio_train'
    # test_path = r'D:\CMZ\dataset\audio_test'
    # file_utils.spliy_train_test(file_path,train_path,test_path,'mp3')


    # Split audio, set split length, overlap, etc.
    # data_path = r'D:\CMZ\dataset_new\audio_without_background'
    # save_path = r'D:\CMZ\dataset_new\audio_without_background_split_nocoverage'
    # files = glob.glob(os.path.join(data_path,'*.mp3'))
    # for file in tqdm(files):
    #     audio_utils.audio_split(file,4,4,save_path)
    # data_path = r'D:\CMZ\dataset_new\audio_without_background_test'
    # save_path = r'D:\CMZ\dataset_new\audio_split\without_background\test_0'
    # files = glob.glob(os.path.join(data_path, '*.mp3'))
    # for file in tqdm(files):
    #     audio_utils.audio_split(file, 4, 4, save_path)

    '''
    
    # Acoustic feature extraction
    # This part is applied to generate the dataset (data.npy, label.csv) in the baseline training code
    # The input is the filefolder of data (train/test), it will produce the .npy file for feature data, and the .csv for label data.
    # Notice that if you want to generate different acoustic features (such as MFCC, Mel, STFT), you need to go to this method and apply different methods of feature extraction
    # X_mfcc,Y_mfcc = audio_utils.preprocessing_classification(audio_utils.dataset_path_train)
    # Y_mfcc.to_csv('train_label_nmfcc400.csv')
    # np.save("train_mfcc_nmfcc400", X_mfcc)
    #
  

