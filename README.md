# Number_Plate_Recognition
In this repository we extract and recognize the plate number from the cars by using Tesseract OCR engine.

Here vehicle number plate is recognize by using opencv, tensorflow tesseract and easyocr engine.
In opencv the plate number is detection by haarcascade files :- https://github.com/lalchhabi/Number_Plate_Recognition/blob/master/haarcascade_russian_plate_number.xml
after the plate number being recognized the text from plate number is extracted by Tesseract OCR engine. To run the program first we need to download haarcascades files 
install and import all the required libraries and locate the path of installed ocr file. Overall project is implemented in app build from streamlit 
https://github.com/lalchhabi/Number_Plate_Recognition/blob/master/app.py

In tensorflow we use pretained model SSD_Mobilenet_V2_fpnlite to detect the plate number. After that the text from plate is extracted by EasyOCR engine.
Here the plate number is detected from image and live webcam as well. When the plate number is detected from images the detected images are stored in 
detected_images and detected text is stored in output.csv.
when the plate number is detected in webcam the detected images are stored in webcam_images
directory and the detected text are stored in live_webcam.csv
