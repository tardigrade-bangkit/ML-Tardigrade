import cv2
import ocr
# loads the input image
image_path = 'atam.jpg'
image = cv2.imread(image_path)
#from google.colab import files
#image = files.upload() 
from google.colab.patches import cv2_imshow
cv2_imshow(image)
prediksi(image)
