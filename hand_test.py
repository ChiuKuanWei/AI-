import cv2
import mediapipe as mp
import threading

# 抓取手掌
def detect_hands(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    return results

# 畫出手掌輪廓
def draw_hand_contour(image, landmarks):
    if landmarks:
        for landmarks in landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

def count_hands(frames,results):
    if results.multi_hand_landmarks:
        draw_hand_contour(frames, results.multi_hand_landmarks)
        return len(results.multi_hand_landmarks)
    else:
        return 0

def process_frames():
    while True:
        # ret:是否在讀取 frame:影像
        ret, frame = cap.read()
        if not ret:
            break

        hand_results = detect_hands(frame)

        # 獲取手部數量
        num_hands = count_hands(frame,hand_results)

        # 在影像上繪製手部數量
        cv2.putText(frame, f'Hands: {num_hands}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示處理後的影像
        cv2.imshow('Hand Contour', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

# 初始化視訊捕獲
cap = cv2.VideoCapture(0)

# 創建一個線程來處理影像
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

# 等待指定執行緒完成，在執行後續動作
processing_thread.join()

if not cap.isOpened():
    # 釋放資源
    cv2.destroyAllWindows()


