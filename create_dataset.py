import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
import av
import numpy as np

from huggingface_hub import hf_hub_download
from transformers import VivitImageProcessor, VivitModel
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

from video_pixel import pixel_prepare
from audio_spectrogram import construct_dataset
from image_pixel import img_pixel_prepare
from extract_ta_feature import extract_feature

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_csv(path):
    with open(path, 'rb') as f:
        return f.readlines()

def get_length(x):
    return x.shape[1] - (np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)

        set_model = config.set_model

        if not set_model:
            # self.train = load_pickle(DATA_PATH + '/train_mosi.pkl')
            # self.valid = load_pickle(DATA_PATH + '/valid_mosi.pkl')
            # self.test = load_pickle(DATA_PATH + '/test_mosi.pkl')
            self.train = load_pickle(DATA_PATH + '/train_mosi_0713.pkl')
            self.valid = load_pickle(DATA_PATH + '/valid_mosi_0713.pkl')
            self.test = load_pickle(DATA_PATH + '/test_mosi_0713.pkl')

        else:

            # self.train = load_pickle(DATA_PATH+'/raw_train_mosi.pkl')
            # self.valid = load_pickle(DATA_PATH+'/raw_valid_mosi.pkl')
            # self.test = load_pickle(DATA_PATH+'/raw_test_mosi.pkl')
            try:
                self.train = load_pickle(DATA_PATH+'/raw_train_mosi.pkl')
                self.valid = load_pickle(DATA_PATH+'/raw_valid_mosi.pkl')
                self.test = load_pickle(DATA_PATH+'/raw_test_mosi.pkl')
                
            except:
                raw_video_path = r'./datasets/MOSI/Raw/Video/Segmented/'
                raw_audio_path = r'./datasets/MOSI/Raw/Audio/WAV_16000/Segmented/'
                csv_filename = r'./datasets/MOSI/texts/mosi/mosi_text.tsv'

                # read csv file for label and text
                data = pd.read_csv(csv_filename,sep='\t',header=0)
                train, valid, test = [], [], []
                train_audio, valid_audio, test_audio = \
                construct_dataset(data, raw_audio_path)
                n_train, n_valid, n_test = 0, 0, 0
                for index, row in data.iterrows():
                    id = row['id']
                    split = row['split']
                    label = row['label']
                    text = row['text']
                    video_path = raw_video_path + '{}.mp4'.format(id)
                    print('video_path:{}'.format(video_path))
                    video_inputs, flag = pixel_prepare(video_path)
                    if flag:
                        # print('create_dataset',video_inputs)
                        if split == 'train':
                            audio_inputs = train_audio[n_train]["audio"]["array"]
                            train.append([(text, video_inputs, audio_inputs),id,label])
                            n_train = n_train + 1
                        elif split == 'valid':
                            audio_inputs = valid_audio[n_valid]["audio"]["array"]
                            valid.append([(text, video_inputs, audio_inputs),id,label])
                            n_valid = n_valid + 1
                        else:
                            audio_inputs = test_audio[n_test]["audio"]["array"]
                            test.append([(text, video_inputs, audio_inputs),id,label])
                            n_test = n_test + 1
                    

                to_pickle(train, DATA_PATH+'/raw_train_mosi.pkl')
                to_pickle(valid, DATA_PATH+'/raw_valid_mosi.pkl')
                to_pickle(test, DATA_PATH+'/raw_test_mosi.pkl')

                self.train = load_pickle(DATA_PATH+'/raw_train_mosi.pkl')
                self.valid = load_pickle(DATA_PATH+'/raw_valid_mosi.pkl')
                self.test = load_pickle(DATA_PATH+'/raw_test_mosi.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class MOSEI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)

        try:
            self.train = load_pickle(DATA_PATH + '/train_mosei.pkl')
            self.valid = load_pickle(DATA_PATH + '/valid_mosei.pkl')
            self.test = load_pickle(DATA_PATH + '/test_mosei.pkl')
        except:
            raw_video_path = r'./datasets/MOSEI/Raw/Video/Segmented/'
            raw_audio_path = r'./datasets/MOSEI/Raw/Audio/WAV_16000/Segmented/'
            csv_filename = r'./datasets/MOSEI/texts/mosei/mosei_text.tsv'

            # read csv file for label and text
            data = pd.read_csv(csv_filename,sep='\t',header=0)
            train, valid, test = [], [], []
            train_audio, valid_audio, test_audio = \
            construct_dataset(data, raw_audio_path)
            n_train, n_valid, n_test = 0, 0, 0
            for index, row in data.iterrows():
                id = row['id']
                split = row['split']
                label = row['label']
                text = row['text']
                video_path = raw_video_path + '{}.mp4'.format(id)
                print('video_path:{}'.format(video_path))
                video_inputs, flag = pixel_prepare(video_path)
                if flag:
                    # print('create_dataset',video_inputs)
                    if split == 'train':
                        audio_inputs = train_audio[n_train]["audio"]["array"]
                        train.append([(text, video_inputs, audio_inputs),id,label])
                        n_train = n_train + 1
                    elif split == 'valid':
                        audio_inputs = valid_audio[n_valid]["audio"]["array"]
                        valid.append([(text, video_inputs, audio_inputs),id,label])
                        n_valid = n_valid + 1
                    else:
                        audio_inputs = test_audio[n_test]["audio"]["array"]
                        test.append([(text, video_inputs, audio_inputs),id,label])
                        n_test = n_test + 1
                    
                to_pickle(train, DATA_PATH+'/train_mosei.pkl')
                to_pickle(valid, DATA_PATH+'/valid_mosei.pkl')
                to_pickle(test, DATA_PATH+'/test_mosei.pkl')

                self.train = load_pickle(DATA_PATH+'/train_mosei.pkl')
                self.valid = load_pickle(DATA_PATH+'/valid_mosei.pkl')
                self.test = load_pickle(DATA_PATH+'/test_mosei.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class IEMOCAP_Causal:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)

        
        self.train = load_pickle(DATA_PATH + '/train_meld_ecf_conver.pkl')
        self.valid = load_pickle(DATA_PATH + '/valid_meld_ecf_conver.pkl')
        self.test = load_pickle(DATA_PATH + '/test_meld_ecf_conver.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class MELD_Causal:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)

        self.train = load_pickle(DATA_PATH + '/train_meld_ecf_conver.pkl')
        self.valid = load_pickle(DATA_PATH + '/valid_meld_ecf_conver.pkll')
        self.test = load_pickle(DATA_PATH + '/test_meld_ecf_conver.pkl')


    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "valid":
            return self.valid
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()
