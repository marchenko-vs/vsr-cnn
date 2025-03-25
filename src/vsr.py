import tkinter
from tkinter import filedialog
import os
import utils
import keras
import vsrutils as vsr
import shutil


# model_filename = '.\\ckpt-mse\\checkpoint-015.keras'
# model = keras.models.load_model(f'.\\{model_filename}', custom_objects={"DepthToSpaceLayer": vsr.DepthToSpaceLayer, 
#                                                                         "ResidualBlock2D": vsr.ResidualBlock2D, 
#                                                                         "psnr": vsr.psnr, "ssim": vsr.ssim})


def get_video_path(textbox):
    filepath = tkinter.filedialog.askopenfilename(initialdir=".", 
        title="Выберите файл", filetypes=(("Файлы MPEG-4", "*.mp4"),))
    
    textbox.delete(0, 'end')
    textbox.insert(0, filepath)


def get_save_path(textbox):
    filepath = tkinter.filedialog.askdirectory(initialdir = ".")

    textbox.delete(0, 'end')
    textbox.insert(0, filepath)


def upscale_video(video_textbox, save_textbox):
    video_path = video_textbox.get()
    save_path = save_textbox.get()

    try:
        os.mkdir('./tmp-lr-frames')
    except Exception:
        pass
    try:
        os.mkdir('./tmp-vsr-frames')
    except Exception:
        pass

    utils.video_to_frames(video_path, './tmp-lr-frames')
    utils.upscale_video(model, './tmp-lr-frames', './tmp-vsr-frames')
    utils.frames_to_video('./tmp-vsr-frames', './upscaled-video.mp4')

    #shutil.rmtree('./tmp-vsr-frames/')
    #shutil.rmtree('./tmp-lr-frames/')


root = tkinter.Tk()
root.geometry('500x300')
root.title('RCNnet')

loadVideoBtn = tkinter.Button(text='Выбрать файл', command=lambda: get_video_path(videoPathTextBox))
loadVideoBtn.grid(column=0, row=0)

videoPathTextBox = tkinter.Entry(width='50')
videoPathTextBox.grid(column=1, row=0)

saveVideoBtn = tkinter.Button(text='Выбрать папку', command=lambda: get_save_path(savePathTextBox))
saveVideoBtn.grid(column=0, row=1)

savePathTextBox = tkinter.Entry(width='50')
savePathTextBox.grid(column=1, row=1)

upscaleVideoBtn = tkinter.Button(text='Увеличить разрешение', command=lambda: upscale_video(videoPathTextBox, savePathTextBox))
upscaleVideoBtn.grid(column=0, row=2)

root.mainloop()

del model
