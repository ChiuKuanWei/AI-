import cv2 
import time
import os

# 設置編號和保存的路徑
save_img_path = './img_lot'
# 如果保存路徑不存在，則建立該路徑
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)

# 開啟webcam
cap = cv2.VideoCapture(0)

# 檢查webcam是否正確開啟
if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()
    
number = 1
while True:
    # 讀取畫面
    ret, frame = cap.read()

    # 如果成功讀取
    if ret:
        # 顯示畫面
        cv2.imshow("Webcam Feed", frame)

        # 檢查鍵盤輸入
        key = cv2.waitKey(1) & 0xFF
        
        # 如果按下 's' 鍵，保存圖片
        if key == ord('s'):
            print('你按下了按鈕,按鈕ASCII碼:',key)
            img_name = f'webcam_img_{number}.jpg'
            cv2.imwrite(os.path.join(save_img_path, img_name), frame) # os.path.join(save_img_path, img_name) = f'./img_lot/webcam_img_{number}.jpg'
            print(f"Saved {img_name}!")
            number += 1

        # 如果按下 'q' 鍵，退出迴圈
        elif key == ord('q'):
            break
    
# 釋放資源並關閉窗口
cap.release()
cv2.destroyAllWindows()