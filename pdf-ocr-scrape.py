# PIL - Pythom Imaging Library
# pytesseract - google optical character recognition library
    # https://pypi.org/project/pytesseract/
    # tesseract-orc - install from:
        # windows: https://github.com/UB-Mannheim/tesseract/wiki
        # others: https://github.com/tesseract-ocr/tesseract/wiki
    # Improve Output quality
        # https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
    # If you don't have tesseract executable in your PATH, include the following:
        # pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
        # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
# OpenCV
    # cv2 - Open Source Computer Vision library:
    # https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26

import pytesseract # must add to path
import PyPDF2
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
import re
import os

from pdf2image import convert_from_path

def OpenImage(c, img):
    x, y, w, h = cv2.boundingRect(c)
    # print(f'y: {y}, h: {h}')
    new_img = img[y:y+h, x:x+w]
    text = str( pytesseract.image_to_string(new_img) ) 
    # Image.fromarray(new_img, 'RGB').show()
    return y, text

TestFiles=[
        'Abbvie Foundation - 2017.pdf',
        'Abbvie Foundation - 2018.pdf',
        'CodeSnippet.JPG',
        'Duke Energy Foundation - 2015.pdf',
        'Exelon Foundation - 2017.pdf',
        'Ford Foundation - 2016.pdf',
        'Mastercard Foundation - 2016.pdf',
        'Page from JPMorgan 990.pdf',
        ]

emptyListPage = []

for file in TestFiles:
    TestFile='PDF-inputs/' + file
    readPDF=PyPDF2.PdfFileReader(TestFile)
    lenPDF=readPDF.getNumPages()
    pages = convert_from_path(TestFile, dpi=200, first_page=0, last_page=lenPDF)
    CurrPage=1

    for page in pages:
        print(f'working on page {CurrPage} of {lenPDF}')
        stringedPage=pytesseract.image_to_string(page)

        # look for wanted pattern
        if bool(re.match(pattern='Form 990PF Part XV Line 3', string=stringedPage)):        

            # get as array and invert sense
            img=np.array(page)[:,:,::-1].copy()
            
            # thresholding the image ?? TODO
            (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            # invert image TODO
            img_bin = 255-img_bin
            # define kernel length TODO
            kernel_length = np.array(img).shape[1]//80
            
            ## detect Horizontal and Vertical lines:
            # Vertical -> vertical kernel of (1 x kernel_length)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
            # Horizontal -> horizontal kernel of (kernel_length X 1)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            # kernel of 3X3 ones
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            
            # Morphological operation to detect vetical lines
            img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
            vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)
            
            # Morphological operation to detect horizontal lines
            img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
            horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=3)
            
            # Weighting parameters - will decide the quantity of an image added to another image
            alpha = 0.5
            beta = 1.0 - alpha
            
            # adds both images with weights to create third image
            img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
            img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)

            # Change to gray scale otherwise contour doesn't work
            # you need a binary image to use findContours(), not a color (bgr) one, use cvtColor()
            img_final_bin = cv2.cvtColor(img_final_bin, cv2.COLOR_BGR2GRAY) 
            thresh, img_final_bin = cv2.threshold(img_final_bin, 127, 255, cv2.THRESH_BINARY) 
            

            # get positions of shapes for later organized extraction
            contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
            emptyList = []
            for c in contours:
                emptyList.append(OpenImage(c, img))

            res = defaultdict(list)
            for k, v in emptyList: res[k].append(v) 
            
            cat2=res.values()
            for caty in cat2:
                emptyListPage.append(caty[::-1]+[CurrPage, file])
        
        CurrPage += 1

PDFtoDF=pd.DataFrame(emptyListPage, columns=['Recipient', 'Relations', 'FoundStatusRec', 'PurposeOfGrant', 'Amount', 'PageNumber', 'File'])
PDFtoDF['Amount']=[re.sub(pattern=',', repl='', string=str(x)) for x in PDFtoDF['Amount']]
filtCol5=~PDFtoDF['Amount'].isin([None, 'Amount', '', 'None'])

PDFtoDFfilt=PDFtoDF[filtCol5]

