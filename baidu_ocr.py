import cv2
from aip import AipOcr

""" 你的 APPID AK SK  """
APP_ID = '15721684'
API_KEY = 'FA3kVbwLFfEQ2BmtLSGa7hyj'
SECRET_KEY = 'XPNs8ZmcSVZxWr5yLiNsqQyq8IyuG33c'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

# fname = 'image/1.jpg'

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content(fname)

""" 调用通用文字识别, 图片参数为本地图片 """
results = client.basicAccurate(image)["words_result"]  # 直接得到字典对应所需字段
print(results)
img = cv2.imread(fname)
for result in results:
    text = result["words"]
    print(text)

# cv2.imwrite(fname[:-4]+"_result.jpg", img)C:\Users\qsp\Desktop\text_recognition\crop_image_use_mouse.py
