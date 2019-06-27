from aip import AipOcr

import numpy
import cv2

from PIL import Image, ImageDraw, ImageFont
import os

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
def put_chinese(image,string,position):
    img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('MSYH.TTC', 25)
    fillColor = (0, 255, 255)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, string, font=font, fill=fillColor)
    img_OpenCV = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV

def text_detect(image_path):
    APP_ID = '15734175'
    API_KEY = 'LqWlaZ29u9MV24fffOIORxAO'
    SECRET_KEY = 'zOOyceFUfUMcLyuiG6pwzsmDAj0CbI7H'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    image_path = get_file_content(image_path)
    result = client.basicAccurate(image_path)#["words_result"]
    if result.__contains__('words_result') and result['words_result_num'] != 0:
        results = result["words_result"]
        text_result = ''
        for result in results:
            text = str(result['words'])
            text_result = text_result+text+' '
        print(text_result)
        #cv2.rectangle(image, (x1 - 5, y1 - 5), (x4 + 5, y4 + 5), (0, 255, 0), 2)
    return text_result
if __name__ =='__main__':
    text_detect("croped.jpg")