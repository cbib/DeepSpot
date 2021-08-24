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
	"conv_block1_filters":32,
	"conv_block2_filters":64,
	"conv_block3_filters":128,
	"conv_block4_filters":128,
	"identity_block_filters":128,
	"upconv_block1_filters":128,
	"epochs":num_epochs,
	"batch_size":batch_size
	}

out_folder_name="/home/ebouilhol/30_papier_acm/results/prediction_arthur/batch_quart_bc"
save_model_folder ="batch_quart_bc"

path1 = "/home/ebouilhol/30_papier_acm/data/arthur/patchs_150/train/original"
path2 = "/home/ebouilhol/30_papier_acm/data/arthur/patchs_150/train/slim"
path3 = "/home/ebouilhol/30_papier_acm/data/arthur/patchs_ultra_clean/test/original"
path4 = "/home/ebouilhol/30_papier_acm/data/arthur/patchs_ultra_clean/test/slim"

path5 = "/home/ebouilhol/30_papier_acm/data/simulated/train_mix_intensities/original"
path6 = "/home/ebouilhol/30_papier_acm/data/simulated/train_mix_intensities/enhanced_slim"
path7 = "/home/ebouilhol/30_papier_acm/data/simulated/train_mix_intensities/original"
path8 = "/home/ebouilhol/30_papier_acm/data/simulated/train_mix_intensities/enhanced_slim"

path_pred_1 = "/home/ebouilhol/30_papier_acm/code_MIRnet/scratch_patchs"
path_pred_2 = "/home/ebouilhol/30_papier_acm/code_MIRnet/scratch_patchs"
path_pred_3 = "/home/ebouilhol/30_papier_acm/code_MIRnet/scratch_patchs"
path_pred_4 = "/home/ebouilhol/30_papier_acm/code_MIRnet/scratch_patchs"

dataset1 = [path1, path2, path3, path4]
dataset2 = [path5, path6, path7, path8]

dataset_pred = [path_pred_1, path_pred_2, path_pred_3, path_pred_4]


out_pred_path="/home/ebouilhol/30_papier_acm/code_MIRnet/scratch_patchs_pred/"

