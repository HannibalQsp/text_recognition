import sys
import os
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont,QPalette
from location_and_ocr_function import *
from crop_image_use_mouse import *
from qr_code import *

class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = os.getcwd() # 获取当前程序文件位置
        self.resize(500,800)   # 设置窗体大小
        self.move(50,100)
        # btn_choose
        self.btn_chooseFile = QPushButton(self)
        self.btn_chooseFile.setObjectName("btn_chooseFile")
        self.btn_chooseFile.setFont(QFont("Microsoft YaHei",10))
        self.btn_chooseFile.setText("选取文件")
        self.btn_chooseFile.move(30,30)
        self.btn_chooseFile.resize(100,30)
        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        # label_file
        self.lable_file = QLabel(self)
        self.lable_file.resize(300,30)
        self.lable_file.move(150,30)
        self.lable_file.setFont(QFont("Times new Roman",12))
        # btn_crop
        self.btn_crop = QPushButton(self)
        self.btn_crop.setObjectName("btn_crop")
        self.btn_crop.setFont(QFont("Microsoft YaHei",10))
        self.btn_crop.setText("截取图片")
        self.btn_crop.move(30,80)
        self.btn_crop.resize(100,30)
        self.btn_crop.clicked.connect(self.crop)
        # btn_detect
        self.btn_detect = QPushButton(self)
        self.btn_detect.setObjectName("btn_detect")
        self.btn_detect.setFont(QFont("Microsoft YaHei",12,))
        self.btn_detect.setText("开始检测")
        self.btn_detect.move(200,130)
        self.btn_detect.resize(100,50)
        self.btn_detect.clicked.connect(self.detect)
        # btn_auto_detect
        self.btn_auto_detect = QPushButton(self)
        self.btn_auto_detect.setObjectName("btn_detect")
        self.btn_auto_detect.setFont(QFont("Microsoft YaHei",12,))
        self.btn_auto_detect.setText("自动检测")
        self.btn_auto_detect.move(250,450)
        self.btn_auto_detect.resize(100,50)
        self.btn_auto_detect.clicked.connect(self.auto_detect)
        # btn_qrcode
        self.btn_qrcode = QPushButton(self)
        self.btn_qrcode.setObjectName("btn_qrcode")
        self.btn_qrcode.setFont(QFont("Microsoft YaHei",12,))
        self.btn_qrcode.setText("二维码识别")
        self.btn_qrcode.move(130,450)
        self.btn_qrcode.resize(100,50)
        self.btn_qrcode.clicked.connect(self.qrcode)
        # btn_image_nor
        self.btn_chooseFile = QPushButton(self)
        self.btn_chooseFile.setObjectName("btn_image_nor")
        self.btn_chooseFile.setFont(QFont("Microsoft YaHei",10))
        self.btn_chooseFile.setText("图像增强")
        self.btn_chooseFile.move(30,140)
        self.btn_chooseFile.resize(100,30)
        self.btn_chooseFile.clicked.connect(self.image_nor)
        # label_result
        self.lable_result = QLabel(self)
        self.lable_result.resize(200,30)
        self.lable_result.move(30,250)
        self.lable_result.setFont(QFont("Microsoft YaHei",12))
        self.lable_result.setText('手动识别结果：')
        # label_result
        self.lable_result = QLabel(self)
        self.lable_result.resize(200,30)
        self.lable_result.move(30,550)
        self.lable_result.setFont(QFont("Microsoft YaHei",12))
        self.lable_result.setText('自动识别结果：')
        # label_text
        self.lable_text = QLabel(self)
        self.lable_text.setWordWrap(True)
        self.lable_text.resize(300,160)
        self.lable_text.move(150,220)
        self.lable_text.setFont(QFont("Microsoft YaHei",12))
        # label_text_auto
        self.lable_text_auto = QLabel(self)
        self.lable_text_auto.setWordWrap(True)
        self.lable_text_auto.resize(300,160)
        self.lable_text_auto.move(150,550)
        self.lable_text_auto.setFont(QFont("Microsoft YaHei",12))
        # label_text_qrcode
        self.label_text_qrcode = QLabel(self)
        self.label_text_qrcode.setWordWrap(True)
        self.label_text_qrcode.resize(300, 120)
        self.label_text_qrcode.move(30, 650)
        self.label_text_qrcode.setFont(QFont("Microsoft YaHei", 12))
    def slot_btn_chooseFile(self):
        self.fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    self.cwd, # 起始路径
                                    "All Files (*);;Text Files (*.txt)")   # 设置文件扩展名过滤,用双分号间隔
        self.lable_file.setText(self.fileName_choose)
    def crop(self):
        crop_image(self.lable_file.text())
    def detect(self):
        self.text = text_detect('croped.jpg')
        print(self.text)
        self.lable_text.setText(self.text)
    def auto_detect(self):
        print(self.lable_file.text())
        image = cv2.imread(self.lable_file.text())
        height= image.shape[0]
        width = image.shape[1]
        croped_image = image[int(0.83*height):height,0:int(0.75*width)] #裁剪固定位置的图片
        cv2.imwrite("croped.jpg",croped_image)
        img = cv2.imread('croped.jpg', 0)
        out = np.zeros(img.shape, np.uint8)
        cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)#图像归一化增强
        cv2.imwrite("croped.jpg", out)
        cv2.imshow("crop",out)
        self.text = text_detect('croped.jpg')
        print(self.text)
        self.lable_text_auto.setText(self.text)
    def image_nor(self):
        img = cv2.imread('croped.jpg', 0)
        out = np.zeros(img.shape, np.uint8)
        cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("croped_image",out)
        cv2.moveWindow("croped_image", 550, 100)
        cv2.imwrite("croped.jpg",out)
    def qrcode(self):
        qrcode_detect(self.lable_file.text())
        self.qrcode_text = qrcode_recognitation()
        self.label_text_qrcode.setText('二维码识别结果：'+self.qrcode_text)
if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('文字识别系统')
    mainForm.show()
    sys.exit(app.exec_())
