from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import shutil
import os
import utils
import vsrutils as vsr
import keras
import glob
import threading
import traceback
import time

model_filename = './_internal/model.keras'
# model_filename = './model.keras'
model = keras.models.load_model(f'.\\{model_filename}', 
                                custom_objects={"DepthToSpaceLayer": vsr.DepthToSpaceLayer, 
                                                "ResidualBlock2D": vsr.ResidualBlock2D, 
                                                "psnr": vsr.psnr, "ssim": vsr.ssim})


@pyqtSlot()
def get_video_path(window, textbox):
    file, _ = QFileDialog.getOpenFileName(window, 'Открыть файл', '.', "Файлы MPEG-4 (*.mp4)")
    textbox.setText(file)


@pyqtSlot()
def get_save_path(window, textbox):
    filepath = QFileDialog.getExistingDirectory(window, 'Сохранить', '.')
    textbox.setText(filepath)


@pyqtSlot()
def upscale_video(window, video_textbox, save_textbox, signal):
    video_path = video_textbox.text()
    save_path = save_textbox.text()

    try:
        os.mkdir('./tmp-lr-frames')
    except Exception:
        pass
    try:
        os.mkdir('./tmp-vsr-frames')
    except Exception:
        pass

    (width, height, count, fps) = utils.video_to_frames(video_path, './tmp-lr-frames')
    inputResolution.setText(f'Исходное разрешение: {width}x{height}')
    outputResolution.setText(f'Увеличенное разрешение: {4 * width}x{4 * height}')
    framesNum.setText(f'Количество кадров: {count}')

    utils.upscale_video(model, './tmp-lr-frames', './tmp-vsr-frames', 5, signal)
    utils.frames_to_video('./tmp-vsr-frames', f'{save_path}/upscaled.mp4', fps=fps)

    shutil.rmtree('./tmp-vsr-frames/')
    shutil.rmtree('./tmp-lr-frames/')
    

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)


class Worker(QRunnable):
    def __init__(self, fn, window, video_textbox, save_textbox):
        super(Worker, self).__init__()

        self.fn = fn
        self.window = window
        self.video_textbox = video_textbox
        self.save_textbox = save_textbox
        self.signals = WorkerSignals()
        self.progress_callback = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(self.window, self.video_textbox, self.save_textbox, self.progress_callback)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def progress_fn(progress: float):
    val = int(100 * progress)
    progressBar.setValue(val)
    progressBar.setFormat(f'{progress:.02f} %')


def _upscale_video(window, video_textbox, save_textbox):
    worker = Worker(upscale_video, window, video_textbox, save_textbox)
    worker.signals.finished.connect(upscaling_finished)
    worker.signals.progress.connect(progress_fn)
    threadpool.start(worker)
    upscaleVideoBtn.setEnabled(False)


def upscaling_finished():
    upscaleVideoBtn.setEnabled(True)


app = QApplication(sys.argv)

window = QWidget()
window.setFixedSize(780, 260)
window.setWindowTitle('RCNnet')

threadpool = QThreadPool()

loadVideoBtn = QPushButton('Выбрать исходный видеофайл', window)
loadVideoBtn.move(10, 20)
loadVideoBtn.resize(240, 30)
loadVideoBtn.setFont(QFont('Times', 12))
loadVideoBtn.clicked.connect(lambda: get_video_path(window, videoPathTextBox))

videoPathTextBox = QLineEdit(window)
videoPathTextBox.move(265, 20)
videoPathTextBox.resize(500, 30)
videoPathTextBox.setFont(QFont('Times', 12))
videoPathTextBox.setReadOnly(True)

saveVideoBtn = QPushButton('Выбрать путь для сохранения', window)
saveVideoBtn.move(10, 60)
saveVideoBtn.resize(240, 30)
saveVideoBtn.setFont(QFont('Times', 12))
saveVideoBtn.clicked.connect(lambda: get_save_path(window, savePathTextBox))

savePathTextBox = QLineEdit(window)
savePathTextBox.move(265, 60)
savePathTextBox.resize(500, 30)
savePathTextBox.setFont(QFont('Times', 12))
savePathTextBox.setReadOnly(True)

upscaleVideoBtn = QPushButton('Увеличить разрешение', window)
upscaleVideoBtn.move(10, 100)
upscaleVideoBtn.resize(240, 30)
upscaleVideoBtn.setFont(QFont('Times', 12))
upscaleVideoBtn.clicked.connect(lambda: _upscale_video(window, videoPathTextBox, savePathTextBox))

progressBar = QProgressBar(window)
progressBar.setGeometry(10, 140, 750, 25)
progressBar.setMaximum(100 * 100)
progressBar.setFont(QFont('Times', 12))

inputResolution = QLabel('Исходное разрешение: ', window)
inputResolution.setFont(QFont('Times', 12))
inputResolution.setGeometry(10, 170, 300, 25)

outputResolution = QLabel('Увеличенное разрешение: ', window)
outputResolution.setFont(QFont('Times', 12))
outputResolution.setGeometry(10, 200, 300, 25)

framesNum = QLabel('Количество кадров: ', window)
framesNum.setFont(QFont('Times', 12))
framesNum.setGeometry(10, 230, 300, 25)

window.show()

app.exec()
