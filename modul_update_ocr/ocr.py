import cv2
import os
import string
import random
from os import listdir
from os.path import isfile, join, splitext
import time
import sys
import numpy as np
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.contours import sort_contours

model_path = 'model_cnn.h5'
model = load_model(model_path)

def prediksi(ob):
  gray = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
  cropped = gray[:,:]
  hasil = cropped.copy()
  for x in range (tinggi):
      for y in range (lebar):
          if cropped[x,y] < 90:
              hasil[x,y] = 0
          else:
              hasil[x,y] = 255

  blurred = cv2.GaussianBlur(hasil, (5, 5), 0)

  
  edged1 = cv2.Canny(blurred, 30, 150) #low_threshold, high_threshold
  edged =  cv2.GaussianBlur(edged1, (5, 5), 0)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")[0]

  chars = []
  for c in cnts:
    # compute the bounding box of the contour and isolate ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = cropped[y:y + h, x:x + w]
    
    #binarize image, finds threshold with OTSU method
    thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # resize largest dimension to input size
    (tH, tW) = thresh.shape
    if tW > tH:
      thresh = imutils.resize(thresh, width=28)
    # otherwise, resize along the height

    # find how much is needed to pad
    (tH, tW) = thresh.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)
    # pad the image and force 28 x 28 dimensions
    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
      left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
      value=(0, 0, 0))
    padded = cv2.resize(padded, (28, 28))
    # reshape and rescale padded image for the model
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)
    # append image and bounding box data in char list
    chars.append((padded, (x, y, w, h)))

  boxes = [b[1] for b in chars]
  chars = np.array([c[0] for c in chars], dtype="float32")
  # OCR the characters using our handwriting recognition model
  preds = model.predict(chars)
  # define the list of label names
  labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

  image = cv2.imread(image_path)
  cropped = image[:,:]
  output=""
  for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    
    output+=label
    # draw the prediction on the image and it's probability
    label_text = f"{label},{prob * 100:.1f}%"
    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0,255 , 0), 2)
    cv2.putText(cropped, label_text, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255, 0), 1)
  # show the image
  plt.figure(figsize=(15,10))
  # plt.imshow(cropped)
  return(output)
