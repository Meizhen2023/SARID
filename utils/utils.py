# -*- coding: utf-8 -*-
# @Time : 2022/11/26 16:20 
# @Author : Mingzheng 
# @File : utils.py
# @desc : tools for other codes
import os
import sys
import errno
import shutil
import json
import os.path as osp
import librosa
from pydub import AudioSegment
import wave
import glob
import pandas as pd
import re
from interval import Interval

from moviepy.editor import *

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def video_clip(file_path,save_path,time_stamp_start,time_stamp_end):
    '''
    切割视频
    :param file_path:视频地址
    :param save_path: 保存地址
    :param time_stamp_start: 起始时间戳（元组形式，（小时，分，秒））
    :param time_stamp_end: 结束时间戳（元组形式，（小时，分，秒））
    :return:
    '''
    # video_clip = CompositeVideoClip([VideoFileClip(file_path).subclip(time_stamp_start,time_stamp_end)])
    start_time_stamp = time_stamp_start[0]*60*60+time_stamp_start[1]*60+time_stamp_start[2]
    end_time_stamp = time_stamp_end[0]*60*60+time_stamp_end[1]*60+time_stamp_end[2]
    video_clip = VideoFileClip(file_path).subclip(start_time_stamp,end_time_stamp)
    video_clip.write_videofile(save_path)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



if __name__ == '__main__':

    file_path = r'D:\CMZ\数据\convert\hiv00034.mp4'
    video_clip(file_path,'hiv00034_2.mp4',(0,22,16),(0,35,30))