import base64
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import cv2
import random


class image(BaseModel):
    image:str

app=FastAPI()

@app.get('/')
def index():

    return {'message': 'This is the homepage of the API '}

@app.post('/image')
def get_image(data:image):
    l=random.choice(range(1, 1000000))
    r=data.dict()
    print(len(r['image']))
    image_64_decode = base64.b64decode(r['image']) 
    image_result = open('./images/image'+str(l)+'.jpg', 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)   
    # Load the cascade  
    face_cascade = cv2.CascadeClassifier('./files/haarcascade_frontalface_default.xml')  
    # Read the input image  
    img = cv2.imread('./images/image'+str(l)+'.jpg')  
    # Convert into grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # Detect faces  
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors= 4)  
  
    # Draw rectangle around the faces  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
  
    # Display the output  
    print(len(faces))
    return{"length":len(faces)}
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
