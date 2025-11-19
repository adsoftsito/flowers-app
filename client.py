import io
import json

import numpy as np
from PIL import Image
import requests
from numpy import asarray
import cv2


#SERVER_URL = "https://flowers-serving-l4tu.onrender.com/v1/models/flowers_model:predict"


SERVER_URL = 'http://localhost:8501/v1/models/flowers-model:predict'

score = 0
image = '/home/codespace/.keras/datasets/flower_photos/flower_photos/tulips/100930342_92e8746431_n.jpg'


def main():
  img = Image.open(image)
  #img = Image.open('dog.png')
  img = img.resize((64, 64))
  img_array = asarray(img)

  #img = np.expand_dims(img_array, 0).tolist()
  #predict_request = json.dumps({'instances': img })
  img_array = img_array.astype(np.float32) / 255.0
  img_array = np.expand_dims(img_array, 0)

  predict_request = {
    "instances": [
        {
            "inputs": img_array.tolist()
        }
    ]
  }
 
  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 1
  index = 0
  for _ in range(num_requests):
    #response = requests.post(SERVER_URL, data=predict_request)
    response = requests.post(SERVER_URL, json=predict_request)

    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions']
#    score = float(sigmoid[0](prediction[0][0]))
    print(prediction[0]) 
    classes_labels = ['dandelion', 'sunflowers', 'daisy', 'tulips', 'roses']
    print(classes_labels)
    image_output_class = classes_labels[np.argmax(prediction[0])]
    print(np.argmax(prediction[0]))

    print("The predicted class is", image_output_class)

#    print(response.json())
#    print ('sigmoid ', sigmoid[0](prediction[0][0]))




if __name__ == '__main__':
  main()