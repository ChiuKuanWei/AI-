#將py檔轉成exe，參考文獻:https://pypi.org/project/auto-py-to-exe/   開啟執行檔指令:"auto-py-to-exe"

import cv2
from PIL import Image

#多行註解快捷鍵 ctrl + /

#設定圖片尺吋
image = Image.open('2.jpg')
new_size =(800,600)
resized_image = image.resize(new_size)
resized_image.save('Output.jpg') #儲存在當前read_image.py檔的路徑上

#讀取圖片
image = cv2.imread('Output.jpg')

if image is not None :
    cv2.namedWindow('show image',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('show image',image)
else:
    print('無法檢視圖片')

# 將圖片轉灰階
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顯示灰階圖片
cv2.namedWindow('Gray Image',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
