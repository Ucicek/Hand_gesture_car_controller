import cv2
import os

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print('Camera did not open. Try again.')
    exit()


labels = {'fullRightDirection': 0,'straightDirection':1,'fullLeftDirection':2,'stopDirection':3}

paths = ['train','test']
for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)

for folder in labels:
    counter = 0

    print(f"For data collection for {folder} press 's' in order to start")
    
    response = input()

    if response != 's':
        print("Wrong input is detected. Exiting.. ")
        break
    
    while counter < 30:
        status,image = camera.read()

        if status == False:
            print('Image capture did not work.')
            break

        cv2.imshow("Video Window", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # if counter > 160:
        #     cv2.imwrite('/Users/utkucicek/Desktop/hand_gesture_based_remote_car_controller/test/'+folder+str(counter)+'_'+str(labels[folder])+'.png',gray)
        # else:
        cv2.imwrite('/Users/utkucicek/Desktop/hand_gesture_based_remote_car_controller/test/'+folder+str(counter)+'_'+str(labels[folder])+'.png',gray)
        counter+=1
        cv2.waitKey(100)
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()




