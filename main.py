from unittest import result
import cv2 
import math
import argparse
def highlightface(net, frame, conf_threashold=0.7):
    frameopencvdnn=frame.copy()
    frameHeight=frameopencvdnn.shape[0]
    frameWidth=frameopencvdnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameopencvdnn,1.0,(300,300),[104,117,120],True,False)
    net.setInput(blob)
    detection=net.forward()
    faceBox=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence > conf_threashold:
            x1=int(detection[0,0,i,3]* frameWidth)
            y1=int(detection[0,0,i,4]* frameHeight)
            x2=int(detection[0,0,i,5]* frameWidth)
            y2=int(detection[0,0,i,6]* frameHeight)
            faceBox.append([x1,y1,x2,y2])
            cv2.rectangle(frameopencvdnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
    return frameopencvdnn, faceBox
parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()
faceProto='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProto='age_deploy.prototxt'
ageModel='age_net.caffemodel'
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'
faceNet=cv2.dnn.readNet(faceModel,faceProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
video=cv2.VideoCapture(args.image if args.image else 0 )
padding=20
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
while cv2.waitKey(1) < 0:
    hasFrame,Frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    resultimg,faceboxes=highlightface(faceNet,Frame)
    if not faceboxes:
        print('No face detected')
    for facebox in faceboxes:
        face=Frame[max(0,facebox[1]-padding):
                    min(facebox[3]+padding,Frame.shape[0]-1),max(0,facebox[0]-padding)
                    :min(facebox[2]+padding, Frame.shape[1]-1)]
        print(face)
        blob=cv2.dnn.blobFromImage(face,1.0,(277,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        genderpredict=genderNet.forward()
        gender=genderList[genderpredict[0].argmax()]
        print(f'Gender:{gender}')
        ageNet.setInput(blob)
        agepredict=ageNet.forward()
        age=ageList[agepredict[0].argmax()]
        print(f'Age:{age[1:-1]}years')
        cv2.putText(resultimg,f'{gender},{age}',(facebox[0],facebox[1]-10),cv2.FONT_HERSHEY_TRIPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('detect age and gender',resultimg)