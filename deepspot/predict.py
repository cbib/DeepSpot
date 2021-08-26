import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import dataset
import skimage.io as io
import time
import pathlib

import global_var as gv

print(tf.version.VERSION)


@tf.function
def test_step(x, y, model):
    pred = model(x, training=False)
    return pred


def custom_loss(y_true, y_pred):
    loss_fn = keras.losses.BinaryCrossentropy()
    bce = loss_fn(y_true, y_pred)
    max_pred = tf.reduce_max(y_true)
    max_true = tf.reduce_max(y_pred)
    mse_max = (max_true - max_pred) ** 2
    loss_value = bce + mse_max
    return loss_value


if __name__ == "__main__":
    ds_pred = dataset.Dataset(gv.dataset_pred)
    pathlib.Path(os.path.join(gv.out_pred_path, 'prediction')).mkdir(parents=True, exist_ok=True)

    tf_ds_test = tf.data.Dataset.from_tensor_slices(ds_pred.dataset_test).batch(gv.batch_size)
    tf_ds_labels_test = tf.data.Dataset.from_tensor_slices(ds_pred.dataset_test_labels).batch(gv.batch_size)
    ds_test = tf.data.Dataset.zip((tf_ds_test, tf_ds_labels_test))

    model = tf.keras.models.load_model( gv.load_model_path, custom_objects={'custom_loss': custom_loss})

    print("Start prediction ")
    start_time = time.time()
    idx = 0
    paths_test = ds_pred.test_list

    for x_batch_val, y_batch_val in ds_test:
        pred = test_step(x_batch_val, y_batch_val, model)
        for i in range(0, len(pred)):
            img_name = os.path.basename(paths_test[idx])
            io.imsave(os.path.join(gv.out_pred_path, "prediction", img_name), pred[i].numpy() * 255,
                      check_contrast=False)
            idx += 1
    print("End prediction")
