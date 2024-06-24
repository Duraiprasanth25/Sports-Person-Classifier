import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

__class_name_to_number = []
__class_number_to_name = []

__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32,32))

        # Flatten the images and concatenate
        scalled_raw_img_flatten = scalled_raw_img.flatten()
        scalled_img_har_flatten = scalled_img_har.flatten()

        combined_img = np.hstack((scalled_raw_img_flatten, scalled_img_har_flatten))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1,len_image_array).astype(float)

        predicted_class_name = __model.predict(final)[0] # this has been changed coz it was showing key error as it was returning string
        if isinstance(predicted_class_name, str):
            predicted_class_number = __class_name_to_number[predicted_class_name]
        else:
            predicted_class_number = predicted_class_name

        # print(f'Predicted class number: {predicted_class_number}')


        predicted_class_name = class_number_to_name(predicted_class_number)
        result.append({
            'class': predicted_class_name,
            'class_probability': np.round(__model.predict_proba(final) *100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })
        
    return result

def load_saved_artifacts():
    print("loading saved artifacts..start")
    global __class_name_to_number
    global __class_number_to_name


    with open("./artifacts/class_dictionary.json","r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

        # print(f"Class name to number mapping: {__class_name_to_number}")
        # print(f"Class number to name mapping: {__class_number_to_name}")

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
        print("loading saved artifacts...done")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_casacde = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    if image_path:  
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_casacde.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces

def get_64_test_image_for_virat():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base64_file_path = os.path.join(current_dir, 'b64.txt')

    with open(base64_file_path) as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(get_64_test_image_for_virat(), None))
    print(classify_image(None, "./test_images/federer1.jpg"))
    print(classify_image(None, "./test_images/federer2.jpg"))
    print(classify_image(None, "./test_images/virat1.jpg"))
    print(classify_image(None, "./test_images/serena2.jpg"))
    print(classify_image(None, "./test_images/virat3.jpg"))
