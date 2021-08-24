#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

num_epochs = 1000
batch_size = 32
batch_size_exp = 24
batch_size_sim = 8
img_size = (256, 256)

config = {
    "learning_rate": 0.0001,
    "dropout_rate": 0.2,
    "conv_block1_filters": 32,
    "conv_block2_filters": 64,
    "conv_block3_filters": 128,
    "conv_block4_filters": 128,
    "identity_block_filters": 128,
    "upconv_block1_filters": 128,
    "epochs": num_epochs,
    "batch_size": batch_size
}

out_folder_name = "path/to/results/"
save_model_folder = "models/Mmixed/"

ds1_train_original = "data/train/original"
ds1_train_GT = "data/train/ground_truth"
ds1_test_original = "data/test/original"
ds1_test_GT = "data/test/ground_truth"
dataset1 = [ds1_train_original, ds1_train_GT, ds1_test_original, ds1_test_GT]

ds2_train_original = "data/train/original"
ds2_train_GT = "data/train/ground_truth"
ds2_test_original = "data/test/original"
ds2_test_GT = "data/test/ground_truth"
dataset2 = [ds2_train_original, ds2_train_GT, ds2_test_original, ds2_test_GT]


# For prediction only
path_pred = "path/to/data"
dataset_pred = [path_pred, path_pred, path_pred, path_pred]
out_pred_path = "path/to/results"
