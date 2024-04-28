from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
app = FastAPI()

class InputData(BaseModel):
    file_path: str

loaded_model = tf.keras.models.load_model('notebooks\\model.keras')
loaded_model.load_weights('notebooks\\best.weights.h5')

@app.get("/predict")
def read_root():
    path = os.path.join("C:\\Users\\wesla\\Desktop", "FS_Surcouf.jpg")
    images = []
    target_size=(224,224)
    img = load_img(path, target_size=target_size)
    img = img_to_array(img)/255
    images.append(img)
    images = np.asarray(images)
    predic = loaded_model.predict(images)
    return predic.tolist()