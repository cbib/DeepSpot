#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

num_epochs = 5
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

save_model_folder = "models/newmodel"

ds1_train_original = "../test_data/dataset1/train/original"
ds1_train_GT = "../test_data/dataset1/train/target"
ds1_test_original = "../test_data/dataset1/test/original"
ds1_test_GT = "../test_data/dataset1/test/target"
dataset1 = [ds1_train_original, ds1_train_GT, ds1_test_original, ds1_test_GT]

ds2_train_original = "../test_data/dataset2/train/original"
ds2_train_GT = "../test_data/dataset2/train/target"
ds2_test_original = "../test_data/dataset2/test/original"
ds2_test_GT = "../test_data/dataset2/test/target"
dataset2 = [ds2_train_original, ds2_train_GT, ds2_test_original, ds2_test_GT]


# For prediction only
load_model_path = "models/Mmixed/"
path_pred = "../test_data/dataset2/test/original"
dataset_pred = [path_pred, path_pred, path_pred, path_pred]
out_pred_path = "results/"
