import os
import json
import pandas as pd
from PIL import Image
import pickle

import numpy as np
import random
import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

from tqdm import tqdm
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Path
data_path = 'C:/Users/nguye/OneDrive/Máy tính/imgcap/COCO-MINI2014'

train_images_path = data_path + '/train_Images/Images'
val_images_path = data_path + '/val_Images/Images'
test_images_path = data_path + '/test_Images'

captions_path = data_path + '/captions.json'

save_path = data_path + '/features_and_weights'
features_path = save_path + '/CNN_encoder_features.pkl'
transformer_weights_path = save_path + '/transformer_decoder_weights.pth'
lstm_weights_path = save_path + '/lstm_decoder_weights.pth'