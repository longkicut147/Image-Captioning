import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torch
from transformers import BertTokenizer
import pandas as pd
import os
import json
from PIL import Image
import random
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# path
# Path
data_path = 'C:/Users/nguye/OneDrive/Máy tính/imgcap/COCO-MINI2014'

train_images_path = data_path + '/train_Images'
val_images_path = data_path + '/val_Images'
test_images_path = data_path + '/test_Images'

captions_path = data_path + '/captions.json'

save_path = data_path + '/features_and_weights'
features_path = save_path + '/CNN_encoder_features.pkl'
weights_path = save_path + '/transformer_decoder_weights.pth'