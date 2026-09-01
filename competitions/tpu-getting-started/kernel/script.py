"""
TPU Getting Started - Petals to the Metal (Flower Classification, 104 classes).

Transfer learning with a pretrained EfficientNet backbone (Keras Applications),
replacing the previous from-scratch 3-layer CNN. This is the standard strong
approach for this competition. Metric: macro F1.

Runs on Kaggle GPU (T4 — NOT P100, which is incompatible with the shipped torch;
TF/Keras is fine on either, but we standardize on T4). Internet must be ON to
download pretrained ImageNet weights.

Data: TFRecords, images at multiple resolutions. We use 192x192.
"""

import glob
import math
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

INPUT = "/kaggle/input/tpu-getting-started"
OUT = "/kaggle/working"
IMAGE_SIZE = [192, 192]
BATCH = 32
EPOCHS = 12
NUM_CLASSES = 104
AUTOTUNE = tf.data.AUTOTUNE

# Locate the 192x192 TFRecord subdir (this comp ships tfrecords-jpeg-{res}).
def find_files(split):
    for res in ("192x192", "224x224", "331x331", "512x512"):
        p = sorted(glob.glob(f"{INPUT}/tfrecords-jpeg-{res}/{split}/*.tfrec"))
        if p:
            return p
    # Fallback: any tfrec matching split
    return sorted(glob.glob(f"{INPUT}/**/{split}/*.tfrec", recursive=True))


print("INPUT listing:", os.listdir(INPUT) if os.path.isdir(INPUT) else "MISSING")
train_files = find_files("train")
val_files = find_files("val")
test_files = find_files("test")
print(f"train={len(train_files)} val={len(val_files)} test={len(test_files)}")
assert train_files and test_files, "No TFRecords found — check INPUT path/attachment"


def count_items(filenames):
    # Filenames embed the record count, e.g. "...-687.tfrec".
    n = 0
    for f in filenames:
        m = re.search(r"-([0-9]+)\.tfrec", os.path.basename(f))
        n += int(m.group(1)) if m else 0
    return n


def decode_image(data):
    # Keras EfficientNet includes its own rescaling/normalization layer, so we
    # feed raw 0-255 float pixels here (dividing by 255 would double-normalize).
    img = tf.image.decode_jpeg(data, channels=3)
    img = tf.cast(img, tf.float32)
    return tf.reshape(img, [*IMAGE_SIZE, 3])


def read_labeled(example):
    feat = {"image": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64)}
    ex = tf.io.parse_single_example(example, feat)
    return decode_image(ex["image"]), tf.cast(ex["class"], tf.int32)


def read_test(example):
    feat = {"image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string)}
    ex = tf.io.parse_single_example(example, feat)
    return decode_image(ex["image"]), ex["id"]


def augment(img, label):
    # Magnitudes scaled for 0-255 pixel range.
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 25.0)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def load(filenames, reader, training=False):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.map(reader, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE).shuffle(2048)
    return ds


n_train = count_items(train_files)
n_val = count_items(val_files)
print(f"n_train={n_train} n_val={n_val}")

train_ds = load(train_files + val_files, read_labeled, training=True).batch(BATCH).prefetch(AUTOTUNE)
steps = math.ceil((n_train + n_val) / BATCH)

# Pretrained backbone (downloads ImageNet weights — needs internet ON).
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="avg")
base.trainable = True

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)),
    base,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])

lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=steps * EPOCHS)
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

print(f"\nTraining {EPOCHS} epochs...")
model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=steps, verbose=2)

print("\nPredicting test...")
test_ds = load(test_files, read_test).batch(BATCH)
ids, preds = [], []
for images, batch_ids in test_ds:
    p = model.predict(images, verbose=0)
    preds.extend(np.argmax(p, axis=1).tolist())
    ids.extend([i.decode("utf-8") for i in batch_ids.numpy()])

sub = pd.DataFrame({"id": ids, "label": preds})
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(f"Saved submission: {sub.shape}")
print("label distribution (top 5):", sub["label"].value_counts().head().to_dict())
