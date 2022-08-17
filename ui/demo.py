from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import json
import sys
from track_mot import onlineTrack


class WorkerThread(QThread):
    def __init__(self):
        super().__init__()
        self.is_exit = False

    def setWidget(self, widget):
        self.widget = widget
        if widget == None:
            self.is_exit = True

    def run(self):
        onlineTrack(self.widget.lomo_config,self.widget.cmb.currentIndex(), self)

    def doRender(self, img):
        if self.widget != None:
            self.widget.doRender(img)


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.initUI()
        self.thread = WorkerThread()
        lomo_config_path = "../lomo/config.json"
        with open(lomo_config_path, "r") as f:
            self.lomo_config = json.load(f)

    def initUI(self):
        self.resize(1280, 960)
        self.setWindowTitle('Soccer Tracker')

        self.layout = QVBoxLayout(self)
        self.video_canvas = QLabel()

        self.layout3 = QHBoxLayout()
        self.lbl1 = QLabel("camera1")
        self.lbl2 = QLabel("camera2")
        self.lbl1.setMaximumHeight(100)
        self.lbl2.setMaximumHeight(100)

        self.layout3.addWidget(self.lbl1)
        self.layout3.addWidget(self.lbl2)

        self.layout2 = QHBoxLayout()


        self.btn = QPushButton("start tracker")
        self.cmb = QComboBox()
        self.cmb.addItem('视频1')
        self.cmb.addItem('视频2')
        self.cmb.addItem('视频3')

        self.layout2.addWidget(self.cmb)
        self.layout2.addWidget(self.btn)

        self.layout.addWidget(self.video_canvas)
        self.layout.addLayout(self.layout3)

        self.layout.addLayout(self.layout2)
        self.setLayout(self.layout)
        self.video_canvas.setMinimumHeight(600)
        self.black_pixmap = QPixmap(self.width(), self.height())
        self.black_pixmap.fill(Qt.black)
        self.video_canvas.setPixmap(self.black_pixmap)
        self.btn.clicked.connect(self.startDetection)
        self.show()

    def startDetection(self):
        if self.thread != None:
            self.thread.setWidget(None)
            self.thread.exit()
        self.thread = WorkerThread()
        self.thread.setWidget(self)
        self.thread.start()

    def doRender(self, img):
        qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.video_canvas.size(), Qt.KeepAspectRatio)
        self.video_canvas.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWidget()
    w.setWindowTitle('Simple')
    w.show()
    sys.exit(app.exec_())
