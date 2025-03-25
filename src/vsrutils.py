import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="VSRLayers")
def psnr(orig, pred):
	orig = orig * 255.0
	orig = tf.cast(orig, tf.uint8)
	orig = tf.clip_by_value(orig, 0, 255)

	pred = pred * 255.0
	pred = tf.cast(pred, tf.uint8)
	pred = tf.clip_by_value(pred, 0, 255)

	return tf.image.psnr(orig, pred, max_val=255)

@register_keras_serializable(package="VSRLayers")
def ssim(orig, pred):
    return tf.image.ssim(orig, pred, max_val=1)


@register_keras_serializable(package="VSRLayers")
class DepthToSpaceLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, factor):
        return tf.nn.depth_to_space(x, factor)
    
    def compute_output_shape(self):
        return (None, None, 1)

    def get_config(self):
        return {}


@register_keras_serializable(package="VSRLayers")
def ssim_loss_fn(orig, pred):
	return 1 - tf.reduce_mean(tf.image.ssim(orig, pred, 1.0))


@register_keras_serializable(package="VSRLayers")
def ResidualBlock3D(inputs):
    x = layers.Conv3D(64, 3, padding="same", activation='relu')(inputs)
    x = layers.Conv3D(64, 3, padding="same")(x)

    return inputs + x


@register_keras_serializable(package="VSRLayers")
def ResidualBlock2D(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)

    return inputs + x
