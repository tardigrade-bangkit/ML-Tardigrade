#import lib
from flask import Flask, render_template, request
import numpy as np
import os
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from tensorflow import keras

#import model
model = keras.models.load_model("model_new.h5")
print("model is loaded")

#flas main appnya gais
app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])

#langsung diarahkan ke index html untuk ngambil gambar
def home():
    return render_template('index.html')

#function predict dijadikan route supaya menangkap gambar dari index
@app.route("/predict", methods=['GET', 'POST'])

#predict
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        test_image = tf.keras.preprocessing.image.load_img(file_path)
        src = cv2.imread(file_path)
        print(src)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inv, (1, 1), 0)
        edged = cv2.Canny(blurred, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        chars = []
        for c in cnts:
          
            (x, y, w, h) = cv2.boundingRect(c)

            if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):

                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape

                if tW > tH:
                    thresh = imutils.resize(thresh, width=32)

                else:
                    thresh = imutils.resize(thresh, height=32)

                (tH, tW) = thresh.shape
                dX = int(max(0, 32 - tW) / 2.0)
                dY = int(max(0, 32 - tH) / 2.0)

                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
                padded = cv2.resize(padded, (32, 32))

                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis=-1)

                chars.append((padded, (x, y, w, h)))
        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")

        preds = model.predict(chars)

        labelNames = "0123456789"
        labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labelNames = [l for l in labelNames]

        output = ""
        for (pred, (x, y, w, h)) in zip(preds, boxes):
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]
            output += label

        print("output",output)

        return render_template('sec.html', pred_output=output, user_image=file_path)


if __name__ == "__main__":
    app.run(threaded=False)