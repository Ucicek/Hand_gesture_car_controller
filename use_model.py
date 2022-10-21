import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import serial
from models import BasicNetwork, ResnetNetwork
from dataload import CustomData


class Prediction:
    def __init__(self,model):
        self.model = model
        self.model.load_state_dict(torch.load('/Users/utkucicek/Desktop/hand_gesture_based_remote_car_controller/model_20221019_020252_3'))
        self.labels = {0:'fullRightDirection', 1:'straightDirection'
            ,2:'fullLeftDirection',3:'stopDirection'}

    def preprocess_image(self,image,size =(112,112)):
        img = Image.fromarray(image).resize(size)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.reshape(img,(1,112,112))
        img = torch.repeat_interleave(img, repeats=3, dim=0)
        return img

    def get_prediction(self,tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, classes = torch.max(probs, 1)
        return conf, classes
    
    def __call__(self,image):
        processed_image = self.preprocess_image(image,size=(112,112))
        processed_image = torch.unsqueeze(processed_image, dim=0)
        return self.get_prediction(tensor=processed_image)

    
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print('Camera did not open. Try again.')
        exit()

    counter = 0

    print("To start controlling the car press 's'")
    
    response = input()

    if response != 's':
        print("Wrong input is detected. Exiting.. ")
        exit()
    labels = {0: 'fullRightDirection', 1: 'straightDirection', 2: 'fullLeftDirection', 3: 'stopDirection'}
    # model = BasicNetwork()
    arduinoData = serial.Serial('port goes here ex. com3',115200)# enter the port
    model = ResnetNetwork()
    preds = Prediction(model)
    isControlling = True
    while isControlling:
        status,image = camera.read()

        if status == False:
            print('Image capture did not work.')
            break
        
        cv2.imshow("Video Window", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        conf,label_number = preds(gray)
        cmd = labels[label_number[0].item()] + '\r' # carriage return
        arduinoData.write(cmd.encode())# sending the code to the arduino
        cv2.waitKey(1500)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

