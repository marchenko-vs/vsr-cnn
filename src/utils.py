import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import PIL
import glob
import os
import cv2
import vsrutils as vsr


def video_to_frames(video_path, frames_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    count = 1
    success = True

    success, image = video.read()
    height, width, channels = image.shape

    while success:
        cv2.imwrite(os.path.join(frames_path, "frame-{:04d}.png".format(count)), image)
        success, image = video.read()
        count += 1

    video.release()

    return (width, height, count - 1, fps)
        

def frames_to_video(frames_path, video_path, fps=24): 
    frame_paths = sorted(glob.glob(f'{frames_path}/*.png'))
    frames = list()

    for i in range(len(frame_paths)):
        frame = cv2.imread(frame_paths[i])
        frames.append(frame)
    
    height, width, layers = frames[0].shape
    size = (width, height)

    video = cv2.VideoWriter(os.path.join(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for i in range(len(frames)):
        video.write(frames[i])

    video.release()


def upscale_frame(model, frames: list, window_size=5, channels=1):
    width, height = frames[0].size
    input_vector = np.empty(shape=(1, window_size, height, width, channels))

    for i in range(len(frames)):
        ycbcr = frames[i].convert("YCbCr")
        (y, _, _) = ycbcr.split()
        y = img_to_array(y)
        y = y.astype("float32") / 255.0
        input_frame = np.expand_dims(y, axis=0)
        input_vector[0, i] = input_frame
    
    out_frame = model.predict(input_vector)

    out_img_y = out_frame[0]
    out_img_y *= 255.0

    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    (_, cb, cr) = frames[window_size // 2].convert("YCbCr").split()
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert("RGB")
    
    return out_img


def upscale_video(model, path_in, path_out, window_size=5, signal=None):
    if signal:
        signal.emit(0)

    frames = sorted(glob.glob(os.path.join(path_in, 'frame-*.png')))

    step = 100 / len(frames)
    progress = 0

    for i in range(window_size // 2):
        frames.insert(0, frames[0])
        frames.append(frames[-1])

    for i in range(len(frames) - window_size + 1):
        frames_vector = list()

        for _ in range(window_size):
            x = PIL.Image.open(frames[i])
            frames_vector.append(x)

        upscaled_frame = upscale_frame(model, frames_vector)
        upscaled_frame.save(os.path.join(path_out, 'frame-{:04d}.png'.format(i + 1)))

        progress += step
        if signal:
            signal.emit(progress)

    if signal:
        signal.emit(100)
