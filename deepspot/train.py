#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras as keras
import time
import pathlib

import dataset
import network
import global_var as gv

print(tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE

def get_ds(path_list, batch_size):
	ds_list = dataset.Dataset(path_list)
	ds = tf.data.Dataset.from_tensor_slices(ds_list.dataset).batch(batch_size, drop_remainder=True)
	ds_labels = tf.data.Dataset.from_tensor_slices(ds_list.dataset_labels).batch(batch_size, drop_remainder=True)
	ds = tf.data.Dataset.zip((ds, ds_labels))
	ds = ds.shuffle(5000)
	return ds, ds_list

def custom_loss(y_true,y_pred):
	loss_fn = keras.losses.BinaryCrossentropy()
	bce = loss_fn(y_true, y_pred)
	max_pred = tf.reduce_max(y_true)
	max_true =tf.reduce_max(y_pred)
	mse_max = (max_true - max_pred)**2
	loss_value = bce + mse_max
	return loss_value

def train(ds, ds_val):
	print("Build model")
	model = network.deepSpot(gv.config, input_shape=(256, 256, 1))
	print("End Build model")

	model.compile(
		optimizer = keras.optimizers.Adam(learning_rate = gv.config['learning_rate']),
		loss = custom_loss,
		metrics = ['mse', 'cosine_similarity', 'accuracy']
	)

	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath = gv.save_model_folder,
		monitor="val_loss",
		verbose=1,
		save_best_only=True,
		mode="min",
	)

	earlystopping = keras.callbacks.EarlyStopping(
		monitor="val_loss",
		mode="min",
		verbose=1,
		patience=200,
		restore_best_weights=True,
	)

	callbacks=[checkpoint, earlystopping]

	model.fit(
		ds,
		epochs = gv.num_epochs,
		verbose = 2,
		validation_data = ds_val,
		callbacks=callbacks
	)

if __name__ == "__main__":
	print("Start dataset generation")
	start_time = time.time()

	ds_exp, ds_exp_list = get_ds(gv.dataset1, gv.batch_size_exp)
	ds_sim, _ = get_ds(gv.dataset2, gv.batch_size_sim)

	arr1 = []
	arr2 = []

	for (x1, y1), (x2,y2) in zip(ds_exp,ds_sim):
		img_batch = tf.concat((x1,x2), 0)
		arr1.append(img_batch)
		label_batch = tf.concat((y1,y2), 0)
		arr2.append(label_batch)

	ds_img = tf.data.Dataset.from_tensor_slices(arr1)
	ds_label = tf.data.Dataset.from_tensor_slices(arr2)

	ds = tf.data.Dataset.zip((ds_img, ds_label))
	ds = ds.shuffle(5000)

	ds_val = tf.data.Dataset.from_tensor_slices(ds_exp_list.dataset_test).batch(gv.batch_size)
	ds_val_label = tf.data.Dataset.from_tensor_slices(ds_exp_list.dataset_test_labels).batch(gv.batch_size)
	ds_val = tf.data.Dataset.zip((ds_val, ds_val_label))

	print("end making dataset")
	print("--- %s seconds ---" % (time.time() - start_time))

	train(ds, ds_val)
