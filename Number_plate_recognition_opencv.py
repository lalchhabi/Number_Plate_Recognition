import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract

#### Path of Installed Tesseract on your local computer
tessdata_dir_config = '/usr/share/tesseract-ocr/4.00/tessdata'

# pytesseract.pytesseract.tesseract_cmd = r'\media\chhabilal\301EAB8B1EAB4924\Program Files\Tesseract-OCR\tesseract.exe'

### haarcascade classifier for plate number classification
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

### Function to extract number from images
def num_extract(img):
    # img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_plate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in num_plate:
        ###### Cropping the plate number region
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]

        ##### Image_preprocessing
        kernel = np.ones((1,1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 120, 255,cv2.THRESH_BINARY)


        ##### Using pytesseract OCR engine to extract the text from images
        text_extract = pytesseract.image_to_string(plate, lang = 'eng', config = tessdata_dir_config)
        #### To remove unnecessary characters in extracted text
        text_extract = ''.join(e for e in text_extract if e.isalnum())
        
        #### Drawing a rectangle box and insert the predicted plate number into it.
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y - 40), (x+w, y), (51,51,255), -1)
        cv2.putText(img, text_extract, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0),1,cv2.LINE_AA)

        return plate, img, text_extract

