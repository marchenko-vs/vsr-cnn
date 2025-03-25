import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import vsrutils as vsr


RDB_LAYERS = 4
FACTOR = 4
CHANNELS = 1
WINDOW_SIZE = 7

conv_args = {"activation": "relu",
             "kernel_initializer": "orthogonal",
             "padding": "same"}

input = keras.Input(shape=(WINDOW_SIZE, None, None, CHANNELS))
x = layers.Conv3D(64, 3, **conv_args)(input)
x = layers.Concatenate()([x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6]])
x = layers.Conv2D(64, 3, **conv_args)(x)
for _ in range(RDB_LAYERS):
    x = vsr.ResidualBlock2D(x)
x = layers.Conv2D(64, 3, **conv_args)(x)
x = layers.Conv2D(CHANNELS * (FACTOR * FACTOR), 3, **conv_args)(x)
outputs = vsr.DepthToSpaceLayer()(x=x, factor=FACTOR)
model = keras.Model(input, outputs)

loss_function = vsr.ssim_loss_fn
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[vsr.psnr, vsr.ssim])

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
model.save(f'.\\models\\{timestamp}.keras')
model.summary()

del model
