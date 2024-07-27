import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import VivitImageProcessor
from transformers import ASTModel,AutoProcessor
from transformers import T5Tokenizer, AutoTokenizer

import random

from create_dataset import MOSI, MELD, UPMC, WIKI
from config import init_device

DEVICE, n_gpu = init_device()
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
video_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")


class Causal_Dataset(Dataset):
    def __init__(self, config):
        ## Fetch dataset
        self.dataset = config.dataset
        self.dataset_name = str.upper(self.dataset)
        dataset = globals()[self.dataset_name](config)
        self.data = dataset.get_data(config.mode)
        self.data_len = len(self.data)
        self.set_model = config.set_model

    @property
    def tva_dim(self):
        # print(self.data[0][0])
        # tva or tv modalities
        modality_num = len(self.data[0][0])
        if modality_num == 3:
            "t,v,a"
            if self.set_model:
                common_dim = 768
                return common_dim, common_dim, common_dim
            else:
                return self.data[0][0][0].shape[1],self.data[0][0][2].shape[1],self.data[0][0][1].shape[1]
        elif modality_num == 2:
            "t,v"
            common_dim = 768
            if self.set_model:
                return common_dim, common_dim, 0
            else:
                return common_dim,self.data[0][0][1].shape[1], 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_len


def get_loader(args, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = Causal_Dataset(config)

    print('mode:{}'.format(config.mode))
    # breakpoint()
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    set_audio = config.tva_dim[2] != 0

    if config.mode == 'train':
        args.n_train = len(dataset)
    elif config.mode == 'valid':
        args.n_valid = len(dataset)
    elif config.mode == 'test':
        args.n_test = len(dataset)
    
    dataset_name = args.dataset

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length

        for sample in batch:
            batch = sorted(batch, key=lambda x: len(x[0][0]), reverse=True)

            v_lens = []
            a_lens = []
            t_lens = []
            label_batch = []
            ids = []

            for sample in batch:
                if len(sample[0])==3:
                    t_lens.append(torch.IntTensor([len(sample[0][0])]))
                    v_lens.append(torch.IntTensor([len(sample[0][1])]))
                    a_lens.append(torch.IntTensor([len(sample[0][2])]))
                elif len(sample[0])==2:
                    # print('visual:{}'.format(sample[0][1]))

                    t_lens.append(torch.IntTensor([len(sample[0][0])]))
                    v_lens.append(torch.IntTensor([len(sample[0][1])]))               
                ids.append(sample[1])
                # print('label:{}'.format(sample[2]))
                # if dataset_name == 'upmc':
                #     label_batch.append(upmc_dict[sample[2]])
                # else:
                label_batch.append(sample[2])


            tlens = torch.cat(t_lens)
            vlens = torch.cat(v_lens)            

            # Rewrite this
            def pad_sequence(sequences, target_len=-1, batch_first=True, padding_value=0.0):

                if target_len < 0:
                    max_size = sequences[0].size()
                    trailing_dims = max_size[1:]
                else:
                    max_size = target_len
                    trailing_dims = sequences[0].size()[1:]
                max_len = max([s.size(0) for s in sequences])
                if batch_first:
                    out_dims = (len(sequences), max_len) + trailing_dims
                else:
                    out_dims = (max_len, len(sequences)) + trailing_dims

                out_tensor = sequences[0].new_full(out_dims, padding_value)
                for i, tensor in enumerate(sequences):
                    length = tensor.size(0)
                    # use index notation to prevent duplicate references to the tensor
                    if batch_first:
                        out_tensor[i, :length, ...] = tensor
                    else:
                        out_tensor[:length, i, ...] = tensor
                return out_tensor

            # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
            visual = pad_sequence([torch.FloatTensor(sample[0][1].cpu()) for sample in batch], target_len=vlens.max().item())
            if len(a_lens) != 0:
                alens = torch.cat(a_lens)
                acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch],target_len=alens.max().item())
            else:
                acoustic = None
                alens = None

            textual = pad_sequence([torch.FloatTensor(sample[0][0].cpu()) for sample in batch], target_len=tlens.max().item())
            labels = torch.LongTensor(label_batch)

            return textual, acoustic, visual, tlens, alens, vlens, labels
        
    def collate_fn_with_pre(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        sampling_rate = 16000

        text_batch = []
        audio_batch = []
        video_batch = []
        label_batch = []
        task_prefix = "sst2 sentence: "
        for sample in batch:
            if len(sample[0])==3:
                (text, video_inputs, audio_inputs), id, label = sample
                audio_batch.append(audio_inputs)
            elif len(sample[0])==2:
                (text, video_inputs), id, label = sample
            text_batch.append(text)
            if dataset_name == 'mosi': # video processor
                pixel_inputs = video_processor(list(video_inputs), return_tensors="pt")
            else: # image processor feature
                pixel_inputs = video_inputs

            video_batch.append(pixel_inputs['pixel_values'])
            label_batch.append(label)

        encoding = tokenizer(
            [task_prefix + sequence for sequence in text_batch],
            return_tensors="pt", padding=True
        )
        # print(label_batch)
        # T5 model things are batch_first
        t5_input_id = encoding.input_ids
        t5_att_mask = encoding.attention_mask
        # target_encoding = tokenizer(
        # label_batch, padding="longest")
        # t5_labels = target_encoding.input_ids
        # t5_labels = torch.tensor(t5_labels)
        # t5_labels[t5_labels == tokenizer.pad_token_id] = -100
        # breakpoint()
        # print(audio_batch)
        video_inputs = torch.cat(video_batch,dim=0)

        labels = torch.LongTensor(label_batch)
        if set_audio:
            audio_inputs = audio_processor(audio_batch, sampling_rate=sampling_rate, return_tensors="pt")
        else:
            audio_inputs = None
        # print('video_inputs.shape:{}'.format(video_inputs.shape))
        return t5_input_id, t5_att_mask, audio_inputs, video_inputs, labels

    if not args.set_model:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_with_pre)
    
    return data_loader
                