# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets,QtGui, uic, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

# import module
import cv2
from keras.models import load_model
import numpy as np
import sys
import os
import time

#webcam
import urllib
from urllib import request

global img_resized
img_resized = any

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() 
        uic.loadUi('gui.ui', self)
        # inisialisasi button

        self.btn_start.clicked.connect(self.start_webcam)
        self.btn_exit.clicked.connect(self.close)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_open.clicked.connect(self.open_video)
        self.btn_image.clicked.connect(self.open_image)

    #timer
    def countdown(time_sec):
        while time_sec:
            mins, secs = divmod(time_sec, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            #print(timeformat, end='\r')
            time.sleep(1)
            time_sec -= 1

    print("timer end")

    #fungsi utama
    def start_webcam(self):
        model = load_model('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/model-020.model')

        face_clsfr=cv2.CascadeClassifier('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/haarcascade_frontalface_default.xml')

        url ='http://192.168.100.81:8080/shot.jpg'

        labels_dict={0:'MASKER',1:'TANPA MASKER'}
        color_dict={0:(0,255,0),1:(0,0,255)}

        print(str(model))
        print(str(face_clsfr))

        try:
            while(True):
                #get video
                imgResp = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
                img_cam = cv2.imdecode(imgNp,-1)
                #preprocessing video
                img=cv2.resize(img_cam,(600,400), fx=0,fy=0, interpolation = cv2.INTER_CUBIC)                
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                faces=face_clsfr.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:                   
        
                    face_img=gray[y:y+w,x:x+w]
                    resized=cv2.resize(face_img,(100,100))
                        
                    normalized=resized/255.0
                    reshaped=np.reshape(normalized,(1,100,100,1))
                    result=model.predict(reshaped)

                    label=np.argmax(result,axis=1)[0]
                    
                    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    img_resized=cv2.resize(img,(600,400))      
                
                cv2.imshow('test',img_resized)
                key=cv2.waitKey(1)
                if(key==27):
                    break
        except:
            print('Kamera Tidak Terdeteksi')
            self.show_warning_cam()

    def open_video(self, file_name):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getOpenFileName(self,"Buka Video", "","mp4 (*.mp4)", options=options)
        file_name_in = os.path.basename(filename[0])
        file_name = str(file_name_in)
        print('successful')
        print(file_name)
        model = load_model('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/model-020.model')

        face_clsfr=cv2.CascadeClassifier('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/haarcascade_frontalface_default.xml')

        source=cv2.VideoCapture(file_name)
        #source=file_name
        #source1 = cv2.imread(source1)
        #source = cv2.cvtColor(source1, cv2.COLOR_RGB2BGR)

        labels_dict={0:'MASKER',1:'TANPA MASKER'}
        color_dict={0:(0,255,0),1:(0,0,255)}
        try:
            while(True):

                ret,img=source.read()
                #img=cv2.imread(source)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_clsfr.detectMultiScale(gray,1.3,5)

                for (x,y,w,h) in faces:

                    time_sec = 3
                    mins, secs = divmod(time_sec, 60)
                    timeformat = '{:02d}:{:02d}'.format(mins, secs)
                    #print(timeformat, end='\r')
                    time.sleep(1)

                    face_img=gray[y:y+w,x:x+w]
                    resized=cv2.resize(face_img,(100,100))
                    normalized=resized/255.0
                    reshaped=np.reshape(normalized,(1,100,100,1))
                    result=model.predict(reshaped)

                    label=np.argmax(result,axis=1)[0]
                    
                    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    img_resized=cv2.resize(img,(600,400))
 
                
                #self.show_video.setPixmap(QPixmap.fromImage(img_resized))
                cv2.imshow('Haar Cascade - Face Mask Detection',img_resized)
                key=cv2.waitKey(1)
                if(key==27):
                    break
        except:
            print('Tidak ada Wajah Terdeteksi!')
            self.show_warning()

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getOpenFileName(self,"Buka File Gambar", "","jpg (*.jpg)", options=options)
        file_name_in = os.path.basename(filename[0])
        file_name = str(file_name_in)
        print('successful')
        print(file_name)
        model = load_model('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/model-020.model')

        face_clsfr=cv2.CascadeClassifier('C:/Users/Microsoft/Documents/Michael/Kampus/semester 5/Comvis/TUGAS UAS/Program UAS/haarcascade_frontalface_default.xml')

        #source=cv2.VideoCapture(file_name)
        source=file_name
        #source1 = cv2.imread(source1)
        #source = cv2.cvtColor(source1, cv2.COLOR_RGB2BGR)

        labels_dict={0:'MASKER',1:'TANPA MASKER'}
        color_dict={0:(0,255,0),1:(0,0,255)}
        try:
            while(True):

                #ret,img=source.read()
                img=cv2.imread(source)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_clsfr.detectMultiScale(gray,1.3,5)
                name=0

                for (x,y,w,h) in faces:
        
                    face_img=gray[y:y+w,x:x+w]
                    resized=cv2.resize(face_img,(100,100))
                    save_image = 'save_image' + str(name) + '.jpg'
                    cv2.imwrite(save_image, resized)
                    normalized=resized/255.0
                    reshaped=np.reshape(normalized,(1,100,100,1))
                    result=model.predict(reshaped)

                    label=np.argmax(result,axis=1)[0]
                
                    cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                    cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                    cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    img_resized=cv2.resize(img,(600,400))
                    name += 1

                #self.show_video.setPixmap(QPixmap.fromImage(img_resized))
                cv2.imshow('Haar Cascade - Face Mask Detection',img_resized)
                key=cv2.waitKey(1)
                if(key==27):
                    break
        except:
            print('Tidak ada Wajah Terdeteksi!')
            self.show_warning()

    #fungsi lain
    def show_warning(self):
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText('Tidak ada Wajah Terdeteksi!')
        msg.setIcon(QMessageBox.Critical)
        x = msg.exec_()

    def show_warning_cam(self):
        msg = QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText('Kamera Tidak Terdeteksi!')
        msg.setIcon(QMessageBox.Critical)
        x = msg.exec_()

    def stop_video(self):
        print('Stopping')
        cv2.destroyAllWindows()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Ui()
    mainWin.show()
    sys.exit( app.exec_() )