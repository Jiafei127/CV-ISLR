#coding:utf-8
import os
import pickle
import zipfile
from os.path import basename
import tqdm
import numpy as np

def read_numpy(file_path):
    return np.load(file_path)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data

with open('../data/label_map.txt', 'r') as f:
    ids = [line.strip() for line in f.readlines()]

def get_prob_labels(file_path):
    pred_meta = load_pkl(file_path)
    out_probs = [x['pred_score'].cpu().numpy() for x in pred_meta]
    out_probs = np.vstack(out_probs) # [51440, 3215]
    out_labels = np.argmax(out_probs, axis=1) # [51440]
    return out_probs, out_labels

# RGB: 
vswin_small_rgb_prob, _ = get_prob_labels('./vswin_small_rgb/result_swin_small_rgb.pkl')
vswin_base_rgb_prob, _ = get_prob_labels('./vswin_base_rgb/result_swin_base_rgb.pkl')
vswin_large_rgb_prob, _ = get_prob_labels('./vswin_large_rgb/result_swin_large_rgb.pkl')
rgb_prob = 0.4 * vswin_large_rgb_prob + 0.4 * vswin_base_rgb_prob + 0.2 * vswin_small_rgb_prob

# Depth: 
vswin_small_depth_prob, _ = get_prob_labels('./vswin_small_depth/result_swin_small_depth.pkl')
vswin_base_depth_prob, _ = get_prob_labels('./vswin_base_depth/result_swin_base_depth.pkl')
vswin_large_depth_prob, _ = get_prob_labels('./vswin_large_depth/result_swin_large_depth.pkl')

depth_prob = 0.65 * vswin_large_depth_prob + 0.35 * vswin_base_depth_prob

fused_prob = rgb_prob * 0.5 + depth_prob * 0.5  # [51440. 3215]

out_labels = np.argmax(fused_prob, axis=1) # [51440]
############################################

with open('./answer.txt', 'w') as f:
    for label in out_labels:
        f.write(ids[label]+'\n')
    f.close()

csv_file_path = './answer.txt'
zip_file_path = './answer.zip'

with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(csv_file_path, os.path.basename(csv_file_path))

