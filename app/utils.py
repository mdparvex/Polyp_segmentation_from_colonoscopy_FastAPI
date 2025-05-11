from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import tensorflow as tf

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask.astype(np.uint8)

def parse_image(file, image_size):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_rgb is None:
        raise ValueError("Unable to decode image")

    image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = image_rgb / 255.0
    return image_rgb

def model_predict(file, model):
    image = parse_image(file, 256)
    predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
    predict_mask = (predict_mask > 0.5) * 255.0
    predict_mask = mask_to_3d(predict_mask)

    # Convert NumPy mask to PIL Image
    mask_image = Image.fromarray(predict_mask)

    # Encode image to base64 string
    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")
    encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_mask}"