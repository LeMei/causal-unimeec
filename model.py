# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label
from operator import contains

#torch.set_printoptions(profile="full")
from dataloader import DEVICE
from models.model.modules.learners.bert_base_model import BertModel

# from warpctc_pytorch import CTCLoss
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# -*- encoding:utf-8 -*-

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import L1Loss, CrossEntropyLoss
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from models.model.logger import MetricLogger

from gnn import GAT

from transformers import RobertaModel, RobertaTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers import VivitModel,ASTModel,T5EncoderModel,VivitConfig
from transformers import VideoMAEConfig, VideoMAEModel, ViTModel

metric_logger = MetricLogger()

class Causal_Prompt_Encoder(nn.Module):
    def __init__(self, args, task_config=None):
        super().__init__()

        # breakpoint()
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.causal_encoder = BertModel.from_pretrained(
                "bert-base-uncased", config=bert_config
        )

    def forward(self, causal_des, x_ids, m1_ids, m2_ids, audio=None, vision=None, text=True):
        
        def index_hidden(ids, hidden):
            hidden_list = []
            for id in ids:
                hidden_list.append(hidden[:,id])

            return torch.cat(hidden_list, dim=-1).to(DEVICE)
            
        def insert_hidden(ids, sequences, padding_value, batch_first=True):
            out_tensor = sequences.new_full(sequences.size(-1), padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, ids[i]:length] = tensor
                else:
                    out_tensor[ids[i]:length, i] = tensor
            return out_tensor
        
        if text:
            causal_text = self.causal_encoder(input_ids=causal_des).last_hidden_state
            x_hid = index_hidden(x_ids, causal_text)
            m1_hid = index_hidden(m1_ids, causal_text)
            m2_hid = index_hidden(m2_ids, causal_text)
            return causal_text, x_hid, m1_hid, m2_hid
        else:
            if not text and not (audio and vision):
                return None

            if audio:
                causal_audio = insert_hidden(x_ids, causal_des, audio)
                causal_audio = self.causal_encoder(query_embeds=causal_audio, 
                                                   return_dict=True).last_hidden_state

                x_hid = index_hidden(x_ids, causal_audio)
                m1_hid = index_hidden(m1_ids, causal_audio)
                m2_hid = index_hidden(m2_ids, causal_audio)
                return causal_audio, x_hid, m1_hid, m2_hid

            if vision:
                causal_vision = insert_hidden(x_ids, causal_des, vision)
                causal_vision = self.causal_encoder(query_embeds=causal_vision, 
                                                   return_dict=True).last_hidden_state

                x_hid = index_hidden(x_ids, causal_vision)
                m1_hid = index_hidden(m1_ids, causal_vision)
                m2_hid = index_hidden(m2_ids, causal_vision)

                return causal_vision, x_hid, m1_hid, m2_hid

class Model(nn.Module):

    def __init__(
            self,
            args,
            task_config=None,
    ):
        super().__init__()

        def set_params_for_layers(name, fine_layers):
            if len(fine_layers) == 0:
                return False

            for task_param in fine_layers:
                if task_param in name:
                    return True
                
        self.set_audio = args.d_ain != 0
        
        if task_config:
            self.task_name = task_config.task_name
            self.dataset_name = task_config.dataset_name
            self.task_dim = task_config.task_dim
            self.task_pred = task_config.task_pred
            self.dropout = task_config.dropout
            self.classifier_dropout = task_config.classifier_dropout
            # print(task_config.task_loss)
            self.task_loss = task_config.task_loss()
            self.pair_strategy = task_config.pair_strategy
            self.fine_learner_layers = task_config.fine_learner_layers
            print('task_name:{}'.format(self.task_name))
            assert self.task_name in [
            "erc", "ecpe"]
            assert self.dataset_name in [
            "iemocap", "meld", "convecpe", "ecf"]
        else:
            print('please offer task name')
            exit()
        
        if args.set_model:
            visual_dim, text_dim, audio_dim = args.d_vin, args.d_tin, args.d_ain
            print('text_dim:{}, audio_dim:{}, visual_dim:{}'.format(text_dim, audio_dim,visual_dim))
            common_dim = 768
           

            self.tokenizer = self.init_tokenizer()

            self.visual_encoder = self.init_vision_encoder(model_name='vit')
            self.audio_encoder = self.init_audio_encoder(model_name='ast')

            self.vision_proj = nn.Linear(visual_dim, common_dim)
            self.text_proj = nn.Linear(text_dim, common_dim)
        else:
            visual_dim, text_dim, audio_dim = args.d_vin, args.d_tin, args.d_ain
            print('text_dim:{}, audio_dim:{}, visual_dim:{}'.format(text_dim, audio_dim, visual_dim))

            common_dim = self.task_dim
            self.vision_proj = nn.Linear(visual_dim, common_dim)
            self.text_proj = nn.Linear(text_dim, common_dim)
            self.audio_proj = nn.Linear(audio_dim, common_dim)


        self.text_fine_layers = task_config.text_fine_layers
        self.av_fine_layers = task_config.av_fine_layers
        self.causal_prompt = Causal_Prompt_Encoder(args, task_config)

        self.m1_gnn_encoder = GAT(common_dim, common_dim,self.task_config.gnn_dropout, \
            self.task_config.alpha,self.task_config.n_heads)
        
        self.m2_gnn_encoder = GAT(common_dim, common_dim, self.task_config.gnn_dropout, \
        self.task_config.alpha,self.task_config.n_heads)

        self.x_gnn_encoder = GAT(common_dim, common_dim,self.task_config.gnn_dropout, \
        self.task_config.alpha,self.task_config.n_heads)
    
        self.set_model = args.set_model
        self.dataset = args.dataset

        self.window = args.window


        self.project = nn.Linear(2*common_dim, common_dim)
        "downstream tasks"
    


        self.emotion_classifier = nn.Sequential(
        nn.Dropout(p=self.classifier_dropout),
        nn.Linear(common_dim, common_dim),
        nn.ReLU(),
        nn.Linear(common_dim, self.task_pred),
        nn.Sigmoid())

        self.cause_classifier = nn.Sequential(
        nn.Dropout(p=self.classifier_dropout),
        nn.Linear(common_dim, common_dim),
        nn.ReLU(),
        nn.Linear(common_dim, self.task_pred),
        nn.Sigmoid())

        self.emotion_loss = nn.CrossEntropyLoss()
        self.cause_loss = nn.CrossEntropyLoss()

    def forward(self, textual, acoustic, visual, erc_label, ecpe_label, device, is_pretrain=False, is_train=True, tlens=None, alens=None, vlens=None):
        # print('textual.shape:{}, visual.shape:{}'.format(textual.shape, visual.shape))
        text_causal, text_x, text_m1, text_m2 = self.causal_prompt(textual, text=True)

        with torch.no_grad():
            visual = self.visual_encoder(visual).pooler_output
            acoustic = self.audio_encoder(acoustic['input_values']).pooler_output

        "unimodal, paired, and modal feature have the same shape after partition"
        visual = self.vision_proj(visual)
        acoustic = self.audio_proj(acoustic)

        visual_causal, visual_x, visual_m1, visual_m2 = self.causal_prompt(textual, vision=visual)
        acoustic_causal, acoustic_x, acoustic_m1, acoustic_m2 = self.causal_prompt(textual, audio=acoustic)

        x_hidden = torch.cat([text_x, visual_x, acoustic_x], dim=-1)
        m1_hidden = torch.cat([text_m1, visual_m1, acoustic_m1], dim=-1)
        m2_hidden = torch.cat([text_m2, visual_m2, acoustic_m2], dim=-1)

        def construct_adj_m1(m1, m2, x, w):
            m1_len, m2_len, x_len = m1.shape[1], m2.shape[1], x.shape[1]
            node_len = m1_len + m2_len + x_len
            adj = torch.zeros([node_len, node_len]).to(DEVICE)
            for i in range(node_len):
                for j in range(node_len):
                    if abs(i-j) <= w:
                        adj[i][j] = 1

            return adj

        def construct_adj_m2(m2, x, w):
            m2_len, x_len = m2.shape[1], x.shape[1]
            node_len =  m2_len + x_len
            adj = torch.zeros([node_len, node_len]).to(DEVICE)
            for i in range(node_len):
                for j in range(node_len):
                    if abs(i-j) <= w:
                        adj[i][j] = 1

            return adj

        def construct_adj_x(x, w):
            x_len = x.shape[1]
            node_len =  x_len
            adj = torch.zeros([node_len, node_len]).to(DEVICE)
            for i in range(node_len):
                for j in range(node_len):
                    if abs(i-j) <= w:
                        adj[i][j] = 1

            return adj
        
        adj_x, adj_m2, adj_m1 = construct_adj_x(x_hidden,self.window), \
        construct_adj_m2(m2_hidden, x_hidden, self.window), \
        construct_adj_m1(m1_hidden, m2_hidden, x_hidden, self.window)

        x_hidden = self.x_gnn_encoder(x_hidden, adj_x)
        m2_hidden = self.m2_gnn_encoder(torch.cat([x_hidden,m2_hidden],dim=1), adj_m2)
        m1_hidden = self.m1_gnn_encoder(torch.cat([m1_hidden, x_hidden,m2_hidden],dim=1), adj_m1)


        emotion_pred = self.emotion_classifier(m1_hidden)
        cause_pred = self.emotion_classifier(m2_hidden)

        emotion_train_loss = self.emotion_loss(emotion_pred, erc_label)
        cause_train_loss = self.cause_loss(cause_pred, ecpe_label)

        return emotion_train_loss, cause_train_loss

        













        