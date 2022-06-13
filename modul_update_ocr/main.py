import ocr
# loads the input image
image_path = 'apel.jpg'
daun = image_path
image = cv2.imread(image_path)
tinggi, lebar = image.shape[:2]
ocr.prediksi(image)
