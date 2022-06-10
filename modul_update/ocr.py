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

def prediksi(ob):
  gray = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
  cropped = gray[10:,10:]
  blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

  # %matplotlib inline
  # from matplotlib import cm
  # fig = plt.figure(figsize=(16,4))
  # ax = plt.subplot(1,4,1)
  # ax.imshow(image)
  # ax.set_title('original image');

  # ax = plt.subplot(1,4,2)
  # ax.imshow(gray,cmap=cm.binary_r)
  # ax.set_axis_off()
  # ax.set_title('grayscale image');

  # ax = plt.subplot(1,4,3)
  # ax.imshow(cropped,cmap=cm.binary_r)
  # ax.set_axis_off()
  # ax.set_title('cropped image');

  # ax = plt.subplot(1,4,4)
  # ax.imshow(blurred,cmap=cm.binary_r)
  # ax.set_axis_off()
  # ax.set_title('blurred image');
  #plt.imshow(gray,cmap=cm.binary_r)

  # perform edge detection, find contours in the edge map, and sort the
  # resulting contours from left-to-right
  edged = cv2.Canny(blurred, 30, 150) #low_threshold, high_threshold
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")[0]

  # figure = plt.figure(figsize=(7,7))
  # plt.axis('off');
  # plt.imshow(edged,cmap=cm.binary_r);

  chars = []
  # loop over the contours
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

      # plot isolated characters
    # n_cols = 10
    # n_rows = np.floor(len(chars)/ n_cols)+1
    # fig = plt.figure(figsize=(1.5*n_cols,1.5*n_rows))
    # for i,char in enumerate(chars):
    #   ax = plt.subplot(n_rows,n_cols,i+1)
    #   ax.imshow(char[0][:,:,0],cmap=cm.binary,aspect='auto')
    #   #plt.axis('off')
    # plt.tight_layout()

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
