import smtplib, ssl
from time import sleep
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
from email.mime.image import MIMEImage
import shutil
import glob
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import time
from multiprocessing import Process
import os
from flask import Flask, render_template, request, url_for
import io
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from multiprocessing import Process
import shutil
import numpy as np

#path = '/home/pi/Desktop/him/'

app = Flask(__name__)

import io
from picamera.array import PiRGBArray
from picamera import PiCamera
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import RPi.GPIO as GPIO
import time
from multiprocessing import Process
import os
import os.path

def dectect():
    time.sleep(0.1)
    createpath = r'./Dataset/JB'
    dirName = 'TEST'

    if os.path.exists(r'./Dataset/JB/TEST'):
        shutil.rmtree(r'./Dataset/JB/TEST')

    os.mkdir(createpath + "/" + dirName + "/")
    path = '/home/pi/Desktop/almost_final/Dataset/JB/TEST'
    camera = PiCamera()
    camera.resolution = (960, 720)
    camera.framerate = 32 
    camera.contrast = 80
    rawCapture = PiRGBArray(camera, size=(960, 720))

    fast = cv2.FastFeatureDetector_create(190, True, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)  # 916 712
    time.sleep(0.1)

    j = 0

    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        kp = fast.detect(image, None)
        # fast.setNonmaxSuppression(0)
        img2 = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))

        if len(kp) > 0: #Cut Around KeyPoint
            x, y = kp[0].pt
            p = int(x)
            q = int(y)
            # print(p,q)

            w = 256
            h = 256

            a = int(q - h / 2)
            b = int(p - w / 2)

            if a < 0:
                a = 0
            if b < 0:
                b = 0

            output = img2[a:int(q + h / 2), b:int(p + w / 2)]
            time.sleep(0.001)
            print('point', a, b)
            cv2.imwrite(os.path.join(path, 'Test' + str(j).zfill(3) + '.jpg'), output)
            #time.sleep(0.03)  # add
            j += 1

            if j > 150:
                break

        cv2.imshow('Frame', img2)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        rawCapture.truncate(0)

    cv2.destroyAllWindows()


def step1(): #Controll Step Motor with clockwise
    time.sleep(1)
    GPIO.setmode(GPIO.BOARD)

    control_pins = [18, 22, 24, 26]

    for pin in control_pins:
        print(pin)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)
    halfstep_seq = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]]

    for i in range(512):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])

            time.sleep(0.002)
    GPIO.cleanup()


def step2(): #Controll Step Motor with Counter clockwise
    time.sleep(1)
    GPIO.setmode(GPIO.BOARD)

    control_pins = [18, 22, 24, 26]

    for pin in control_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)
    revstep_seq = [
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]]

    for i in range(512):
        for revstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], revstep_seq[revstep][pin])

            time.sleep(0.002)

    GPIO.cleanup()


def servo1(): #Controll Servo Motor
    print('servo start')
    time.sleep(0.3)
    GPIO.setwarnings(False)

    # control = [6.5, 6]

    servo = 12

    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(servo, GPIO.OUT)

    p = GPIO.PWM(servo, 50)  # 50hz frequency

    

    p.start(3)  # starting duty cycle ( it set the servo to 0 degree )

    for x in range(2):
        p.ChangeDutyCycle(5)
        time.sleep(0.5)
        # p.ChangeDutyCycle(3)
        # p.ChangeDutyCycle(4)
        # time.sleep(3)

        p.stop()

    GPIO.cleanup()
    print('servo end')


def servo2():
    GPIO.setwarnings(False)

    # control = [6.5, 6]

    servo = 12

    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(servo, GPIO.OUT)

    p = GPIO.PWM(servo, 50)  # 50hz frequency

    p.start(3)  # starting duty cycle ( it set the servo to 0 degree )

    for x in range(1):
        p.ChangeDutyCycle(2)
        time.sleep(0.5)

        p.stop()

    GPIO.cleanup()




def overdose(): #Check the similar picture and delete
    myPath = '/home/pi/Desktop/almost_final/Dataset/JB/TEST'

    myExt = '*.jpg' # 찾고 싶은 확장자

    counter = 0
    imlist=[]
    last_histo = np.zeros((256, 1), int)

    filist = sorted(glob.glob(os.path.join(myPath, myExt)))

    for j in filist:
        print(j)
        #print(type(glob.glob(os.path.join(myPath, myExt))))
        image = cv2.imread(j)
        height = image.shape[0]
        width = image.shape[1]
        #print(width, height)
        threshold = height * width * 0.5



        if (counter == 0):
            #print("카운트 0")
            im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            last_histo = cv2.calcHist([im], [0], None, [256], [0, 256])
            #cv2.imshow('Detected Cut : frame ' + str(counter), im)
            imlist.append(j)
            counter += 1

            #continue    # 다음 반복으로 넘김
        sum_histo_differ = 0  # 히스토그램 변화
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histo = cv2.calcHist([im], [0], None, [256], [0, 256])

        for i in range(256):  # 히스토그램의 bins 수 만큼 반복
            sum_histo_differ += abs(histo[i] - last_histo[i])  # 히스토그램 변화 값 누적

        # Scene change 검출 시(히스토그램 변화 값이 threshold 값을 넘어갈 경우)
        print(sum_histo_differ)

        #이 이프문안에 들어가는 사진만 보내면 되는 것!
        if (sum_histo_differ > threshold):
            print("Detected scene change at frame %d." % (counter))  # 현재 프레임 번호 출력
            last_histo = histo  # 다음 반복 때 비교하기 위해 현재 히스토그램 저장
            #cv2.imshow('Detected Cut : frame ' + str(counter), im)  # 검출된 컷 출력
            imlist.append(j)

        counter += 1  # 프레임 카운트 증가
        #print("루프 하나 끝")
        #cv2.imshow('junsu', image)

        #key = cv2.waitKey(0)
        
        
    for j in filist:
        if j in imlist:
            print(j, 'include') 
        else:
            print(j, 'not include') #Similar Picture
            os.remove(j)

    print(imlist)

    #print('keras start')
    keras_execute()
    #print('keras end')

def send_an_email():
    print("mail start")
    toaddr = temp  # To id
    me = '2019jbtest@gmail.com'  # your id
    subject = "Detect Complete!"  # Subject

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = toaddr
    msg.preamble = "test "
    # msg.attach(MIMEText(text))
    

    path = './detected/'
    
    jpgfiles = glob.glob(os.path.join(path, "*.jpg"))
    
    for file in jpgfiles:
        fp = open(file, 'rb')
        img = MIMEImage(fp.read())
        fp.close()
        img.add_header('Content-Disposition', 'attachment', filename=file)
        msg.attach(img)

    # os.remove('Test_nolense.jpg')  //삭제하는거
    

    try:
        
        s = smtplib.SMTP('smtp.gmail.com', 587)  # Protocol
        
        s.ehlo()
        
        s.starttls()
        
        s.ehlo()
        s.login(user='^^@gmail.com', password='^^')  # User id & password
        
        # s.send_message(msg)
        s.sendmail(me, toaddr, msg.as_string())
        s.quit()
        remove()
        remove2()

    except:
        print("Error: unable to send email")


def remove():
    shutil.rmtree(r'./detected')


def remove2():
    shutil.rmtree(r'./Dataset/JB/TEST')


def keras_execute():
    from keras.models import load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
    from keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    from keras.preprocessing import image
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array

    batch_size = 1  # 배치사이즈, hanbune mutjang

    test_path = './Dataset/JB'  # Test 폴더 아래 폴더를 하나 더 만들고 그 안에 한번에 무작위로 배치

    test_data = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_data.flow_from_directory(test_path,
                                                   target_size=(128, 128),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle='false')

    loaded_model = load_model("modelyj2.h5") #Put your learned model
    print("Loaded model from disk")

    predict = []
    for i, n in enumerate(test_generator.filenames):
        img_path = "./Dataset/JB/" + str(n)
        print(img_path)
        _img = image.load_img(img_path, target_size=(128, 128))
        img = image.img_to_array(_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        predict.append(loaded_model.predict(img, batch_size=None, steps=1))

        correct = 0
        lense = 0
        nolense = 0

    if os.path.exists(r'./detected'):
        shutil.rmtree(r'./detected')

    os.mkdir('./detected')

    copypath = r'./detected'

    for i, n in enumerate(test_generator.filenames):
        print(n)
        if n.startswith("TEST/") and predict[i][0, 0] > predict[i][0, 1]:
            correct += 1
            lense += 1
            print(predict[i][0, 0], " & ", predict[i][0, 1])

            filedir = "./Dataset/JB/"
            new = filedir + n

            shutil.copy2(new, copypath)
            # predict\lense 폴더에 렌즈라고 탐지한 이미지파일 복사

        if n.startswith(r"TEST/") and predict[i][0, 1] > predict[i][0, 0]:
            correct += 1
            nolense += 1
            print(predict[i][0, 0], " & ", predict[i][0, 1])

    print("정답:", correct, "렌즈:", lense, "노렌즈: ", nolense)


@app.route('/')
@app.route('/main')
def main_get(num=None):
    return render_template('jb.html', num=num)


#The part that executes the defined functions

@app.route('/calculate', methods=['GET'])
def calculate():
    global p1
    global p2
    global temp
    temp = request.args.get('email')

    p1 = Process(target=dectect)
    p1.start()

    p2 = Process(target=step1)
    p2.start()

    p2.join()

    p2 = Process(target=servo1)
    p2.start()
    p2.join()
    p2 = Process(target=step2)
    p2.start()
    p2.join()
    p2 = Process(target=servo2)
    # p1.join()
    p1.terminate()
    p2.start()

    p2.join()

    p2 = Process(target=overdose)
    p2.start()
    p2.join()

    p3 = Process(target=send_an_email)
    p3.start()
    p3.join()
    p2.terminate()

    print('p1, p2 end1')
    return render_template('jb.html', email=temp)

    server.terminate()


# first

if __name__ == '__main__':
    global server
    server = Process(target=app.run)
    server.start()












