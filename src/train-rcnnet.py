import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
import vsrutils as vsr
from datetime import datetime
import math
import glob

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

CHECKPOINT_PATH = ".\\ckpt\\checkpoint-{epoch:03d}.keras"

list_of_files = glob.glob('.\\models\\*.keras')
latest_file = max(list_of_files, key=os.path.getctime)
model = keras.models.load_model(f'.\\{latest_file}', custom_objects={"DepthToSpaceLayer": vsr.DepthToSpaceLayer, 
                                                                     "ResidualBlock2D": vsr.ResidualBlock2D,
                                                                     "ResidualBlock3D": vsr.ResidualBlock3D,
                                                                     "ssim": vsr.ssim, "psnr": vsr.psnr})

FACTOR = 4
CHANNELS = 1
EPOCHS = 30
BATCH_SIZE = 4
WINDOW_SIZE = 7
ROOT_PATH = '.\\vimeo-dataset'

OUT_WIDTH = 448
OUT_HEIGHT = 256
IN_WIDTH = OUT_WIDTH // FACTOR
IN_HEIGHT = OUT_HEIGHT // FACTOR

cp_callback = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, verbose=1)


def gen_labels_vimeo(path):
    labels = list()
    dirs = sorted(glob.glob(f'{path}\\input\\*'))

    for dir in dirs:
        subdirs = sorted(glob.glob(f'{dir}\\*'))
        
        for subdir in subdirs:
            labels.append('\\'.join(subdir.split('\\')[3:]))

    length = len(labels)
    middle = int(math.floor(0.8 * length))
    end = length

    return labels[:middle], labels[middle:end]


LABELS, LABELS_VAL = gen_labels_vimeo(ROOT_PATH)
LIST_IDS = [i for i in range(len(LABELS))]
LIST_IDS_VAL = [i for i in range(len(LABELS_VAL))]


print('=' * 20)
print('DATASET INFO')
print(f'Test set size: {len(LABELS)}')
print(f'Validation set size: {len(LABELS_VAL)}')
print('=' * 20)


class VSRDataGenerator(keras.utils.Sequence):
    def __init__(self, id_list, labels, image_path, window_size=WINDOW_SIZE,
                 batch_size=BATCH_SIZE, channels=CHANNELS):
        self.id_list = id_list
        self.labels = labels
        self.image_path = image_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.channels = channels

    def __len__(self):
        return int(np.floor(len(self.id_list) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.id_list[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.id_list[k] for k in indexes]

        X, y = self._generate_batch(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        np.random.shuffle(self.id_list)

    def _generate_batch(self, list_IDs_temp):
        X = np.empty(shape=(self.batch_size, self.window_size, IN_HEIGHT, IN_WIDTH, self.channels))
        y = np.empty(shape=(self.batch_size, OUT_HEIGHT, OUT_WIDTH, self.channels))

        for i, ID in enumerate(list_IDs_temp):
            X[i, 0] = self._load_img('{}\\low_resolution\\{}\\im1.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 1] = self._load_img('{}\\low_resolution\\{}\\im2.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 2] = self._load_img('{}\\low_resolution\\{}\\im3.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 3] = self._load_img('{}\\low_resolution\\{}\\im4.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 4] = self._load_img('{}\\low_resolution\\{}\\im5.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 5] = self._load_img('{}\\low_resolution\\{}\\im6.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            X[i, 6] = self._load_img('{}\\low_resolution\\{}\\im7.png'.format(self.image_path, self.labels[ID]), (IN_HEIGHT, IN_WIDTH))
            
            y[i] = self._load_img('{}\\target\\{}\\im4.png'.format(self.image_path, self.labels[ID]), (OUT_HEIGHT, OUT_WIDTH))

        return X, y
    
    def _load_img(self, image_path, size):
        orig_img = tf.io.read_file(image_path)
        orig_img = tf.image.decode_png(orig_img, 3)
        orig_img = tf.image.convert_image_dtype(orig_img, tf.float32)
        #orig_img = tf.image.resize(orig_img, size, method="area")
        orig_img_yuv = tf.image.rgb_to_yuv(orig_img)
        (orig_img_y, _, _) = tf.split(orig_img_yuv, 3, axis=-1)

        tf.clip_by_value(orig_img_y, 0.0, 1.0)

        return orig_img_y


gen = VSRDataGenerator(LIST_IDS, LABELS, ROOT_PATH)
gen_val = VSRDataGenerator(LIST_IDS_VAL, LABELS_VAL, ROOT_PATH)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
callbacks = [early_stopping_callback, cp_callback]

model.fit(gen, epochs=EPOCHS, validation_data=gen_val, verbose=1, callbacks=callbacks)

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
model.save(latest_file)

tf.keras.backend.clear_session()

del model
