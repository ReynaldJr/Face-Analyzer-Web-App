import numpy as np
import cv2
import sklearn
import pickle
from django.conf import settings
import os 

STATIC_DIR = settings.STATIC_DIR

# Detecting Faces
face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'./models/deploy.prototxt.txt'), 
                                              os.path.join(STATIC_DIR,'./models/res10_300x300_ssd_iter_140000_fp16.caffemodel'))
# Extracting Features
face_feature_model = cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,'./models/openface.nn4.small2.v1.t7'))

# Emotion Recognition
emotion_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'./models/face_emotion.pkl'), mode='rb'))

# Gender Recogntion
gender_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'./models/face_gender.pkl'), mode='rb'))

# Age Recognition
age_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'./models/face_age.pkl'), mode='rb'))

# Race Recognition
race_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'./models/face_race.pkl'), mode='rb'))

def face_recognizer_pipeline_model(img_path):  
    # Pipelining Models
    img = cv2.imread(img_path)
    image = img.copy()
    h,w = img.shape[:2]

    # Detecting Face
    img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()

    # Machine Learning Results
    ml_results = dict(face_detect_score = [],
                      emotion_name = [],
                      emotion_name_score = [],
                      gender_name = [],
                      gender_name_score = [],
                      race_name = [],
                      race_name_score = [],
                      age_name = [],
                      age_name_score = [],
                      count = []
                     )
    count = 1
    if len(detections) > 0:
        for i, confidence in enumerate(detections[0, 0, :, 2]):
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startx, starty, endx, endy = box.astype(int)

                cv2.rectangle(image, (startx, starty), (endx, endy), (255,199,107))  # Increase line thickness for visibility

                # Extracting Features
                face_roi = img[starty:endy, startx:endx]
                face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0), swapRB = True, crop = True)
                face_feature_model.setInput(face_blob)

                vectors = face_feature_model.forward()

                # Predicting with models

                # Emotion
                emotion_predict = emotion_recognition_model.predict(vectors)[0].capitalize()
                emotion_predict_score = emotion_recognition_model.predict_proba(vectors).max()

                emotion_text = '{} : {:.0f}%'.format(emotion_predict.capitalize(),100*emotion_predict_score)
                cv2.putText(image,emotion_text,(startx,endy+30),cv2.FONT_HERSHEY_PLAIN,1.3,(255,199,107),2)

                # Gender
                gender_predict = gender_recognition_model.predict(vectors)[0]
                gender_predict_score = gender_recognition_model.predict_proba(vectors).max()

                gender_text = '{} : {:.0f}%'.format(gender_predict,100*gender_predict_score)
                cv2.putText(image,gender_text,(startx,endy+65),cv2.FONT_HERSHEY_PLAIN,1.3,(255,199,107),2)

                # Race
                race_predict = race_recognition_model.predict(vectors)[0]
                race_predict_score = race_recognition_model.predict_proba(vectors).max()

                race_text = '{} : {:.0f}%'.format(race_predict,100*race_predict_score)
                cv2.putText(image,race_text,(startx,endy+95),cv2.FONT_HERSHEY_PLAIN,1.3,(255,199,107),2)

                # Age
                age_predict = age_recognition_model.predict(vectors)[0]
                age_predict_score = age_recognition_model.predict_proba(vectors).max()

                age_text = '{} : {:.0f}%'.format(age_predict,100*age_predict_score)
                cv2.putText(image,age_text,(startx,endy+125),cv2.FONT_HERSHEY_PLAIN,1.3,(255,199,107),2)
                
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'), image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg'.format(count)), face_roi)
                
                ml_results['count'].append(count)
                ml_results['face_detect_score'].append(confidence)
                ml_results['emotion_name'].append(emotion_predict)
                ml_results['emotion_name_score'].append(emotion_predict_score)
                ml_results['gender_name'].append(gender_predict)
                ml_results['gender_name_score'].append(gender_predict_score)
                ml_results['race_name'].append(race_predict)
                ml_results['race_name_score'].append(race_predict_score)
                ml_results['age_name'].append(age_predict)
                ml_results['age_name_score'].append(age_predict_score)
                
                count += 1
                
    return ml_results