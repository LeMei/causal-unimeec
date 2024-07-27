import pickle
import pandas as pd
import numpy as np
import torch

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_csv(path):
    with open(path, 'rb') as f:
        return f.readlines()

def causal_gen(emotion_utter, emotion_category, cause_id, conversation):

    causal_template = 'In conversation:{}, the emotion of utterance {} is {}, \
    and its emotion cause is {}'

    causal_des = causal_template.format(conversation, emotion_utter, emotion_category, cause_id)

    return causal_template, causal_des


def construct_dataset(erc_data_path, ecpe_data_path, conver_data_path, to_dir):

    erc_data = load_pickle(erc_data_path)
    ecpe_data = load_pickle(ecpe_data_path)
    conver_data = load_pickle(conver_data_path)

    data = []

    for erc_sample, ecpe_sample in zip(erc_data, ecpe_data):
        cid = erc_sample[0] 
        conver = conver_data[cid]
        erc_text, ecpe_text = erc_sample[1], ecpe_sample[1]
        erc_audio, ecpe_audio = erc_sample[2], ecpe_sample[2]
        erc_vision, ecpe_vision = erc_sample[3], ecpe_sample[3]

        erc_label, ecpe_label = erc_sample[4], ecpe_sample[4]

        causal_temp, causal_des = causal_gen(erc_text, erc_label, ecpe_label, conver)

        feature = (cid, causal_des, causal_temp, (erc_audio, ecpe_audio), (erc_vision, ecpe_vision), ecpe_text, erc_label, ecpe_label)

        data.append(feature)

    to_pickle(data, to_dir)


# #### FOR MELD AND ECF

# meld_train_data_path = r'./train_meld.pkl'
# meld_valid_data_path = r'./valid_meld.pkl'
# meld_test_data_path = r'./test_meld.pkl'

# ecf_train_data_path = r'./train_ecf.pkl'
# ecf_valid_data_path = r'./valid_ecf.pkl'
# ecf_test_data_path = r'./test_ecf.pkl'

# meld_erc_to_dir = r'./causal_{}_meld_ecf.pkl'

# conver_train_data_path = r'./train_meld_ecf_conver.pkl'
# conver_valid_data_path = r'./valid_meld_ecf_conver.pkl'
# conver_test_data_path = r'./test_meld_ecf_conver.pkl'

# construct_dataset(meld_train_data_path, ecf_train_data_path, conver_train_data_path, meld_erc_to_dir.format('train'))
# construct_dataset(meld_valid_data_path, ecf_valid_data_path, conver_valid_data_path,meld_erc_to_dir.format('valid'))
# construct_dataset(meld_test_data_path, ecf_test_data_path, conver_test_data_path,meld_erc_to_dir.format('test'))


#### FOR IEMOCAP AND CONVECPE

iemocap_train_data_path = r'./train_iemocap.pkl'
iemocap_valid_data_path = r'./valid_iemocap.pkl'
iemocap_test_data_path = r'./test_iemocap.pkl'

conv_train_data_path = r'./train_conv.pkl'
conv_valid_data_path = r'./valid_conv.pkl'
conv_test_data_path = r'./test_conv.pkl'

iemocap_erc_to_dir = r'./causal_{}_iemocap_conv.pkl'

conver_train_data_path = r'./train_iemocap_conv_conver.pkl'
conver_valid_data_path = r'./valid_iemocap_conv_conver.pkl'
conver_test_data_path = r'./test_iemocap_conv_conver.pkl'

construct_dataset(iemocap_train_data_path, conv_train_data_path, conver_train_data_path, iemocap_erc_to_dir.format('train'))
construct_dataset(iemocap_valid_data_path, conv_valid_data_path, conver_valid_data_path,iemocap_erc_to_dir.format('valid'))
construct_dataset(iemocap_test_data_path, conv_test_data_path, conver_test_data_path,iemocap_erc_to_dir.format('test'))



    