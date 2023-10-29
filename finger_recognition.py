# 搭建模型
from tensorflow.keras.models import Sequential # 模型的蛋糕支柱
from tensorflow.keras.layers import Dense # 描述蛋糕 神經元的細部參數
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import pygame

# 用於引入所需的函式庫和模組
imgs_path = glob.glob('.\img_lot\*.jpg')

# 這個部分使用glob模組通配符來找到以.jpg為擴展名的所有圖片檔案的路徑
# 所有找到的圖片進行處理。將其轉換為灰階，然後將圖片的尺寸調整為30x30。最後，將所有圖片堆疊成一個 NumPy 陣列 x_train
#===============x_train==================
x_train = []
for img_path in imgs_path:
    img = cv2.imread(img_path,0) # cv2.imread(圖片路徑,通道壓縮) 通道壓縮=0 灰階
    # 縮小圖片尺寸
    img_rs = cv2.resize(img,(30,30),interpolation = cv2.INTER_AREA)
    x_train.append(img_rs)

x_train = np.array(x_train)

# 創建一個 DataFrame df，其中包含圖片路徑和對應的答案。filter_path2num 函式將圖片路徑轉換為整數答案，並將這些答案存儲在 y_train 中
#================y_train===================
def filter_path2num(data):
    if data.split('\\')[-1].split('(')[0]=='cutter':
        return int(0)    
    elif data.split('\\')[-1].split('(')[0]=='stone':
        return int(1)
    elif data.split('\\')[-1].split('(')[0]=='paper':
        return int(2)   
    
df = pd.DataFrame(columns = ['paths'])
df['paths'] = imgs_path
# apply(): apply 方法用於應用一個函數到 DataFrame 的每一個元素
df['answer'] = df['paths'].apply(filter_path2num)
y_train = df['answer'].values

# 隨機抽樣看一下數據影像長怎樣
# c = random.randint(0,y_train.shape[0])
# img = x_train[c]
# plt.imshow(img,cmap = 'gray')
# plt.title(f'img_answer:{y_train[c]}')
# plt.show()

# 圖片跟答案要預處理
# 圖片: 要壓到0~1之間(原圖0~255),拉成一條900(30x30)且一維的神經元數量 ; x_train.shape[0]:圖片總數 ; x_train.shape[1]與x_train.shape[2]:分別為寬高
# 將 x_train 的圖片數據重新塑形（reshape）為一維數組，同時將像素值除以 255 來將它們歸一化到範圍 [0, 1]。準備圖片數據以供神經網絡模型訓練時常見的預處理步驟

# 將整數標籤 y_train 轉換為 one-hot 編碼的形式，以便進行多元分類模型的訓練 
#  如，假設有三個類別："狗"、"貓" 和 "鳥"，使用 one-hot 編碼後，它們的表示如下：
#  "狗"：[1, 0, 0]
#  "貓"：[0, 1, 0]
#  "鳥"：[0, 0, 1]
x_train_shape = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255
y_train_cat = to_categorical(y_train)

# print('x圖片資訊:',x_train_shape[c].shape)
# print('的答案資訊:',y_train[c])
# print('one_hot後的答案資訊:',y_train_cat[c])
# one_hot 目的是要將y答案不要有大小之分,在計算loss時比較公平

# 這部分建立了一個 Sequential 模型 mlp，添加了多個全連接層
#region 解說
# 'relu'（整流線性單元）：
# 'relu' 是一種常用的激活函數，用於隱藏層中的神經元。它通常用於增加模型的非線性能力。'relu' 函數將小於零的輸入設置為零，而對於正數輸入，它保持不變。
# 'relu' 的主要作用是引入非線性，使模型能夠學習更複雜的特徵。這有助於模型更好地擬合訓練數據，尤其是在深層神經網絡中。

# 'softmax'：
# 'softmax' 是一種激活函數，通常用於多類別分類問題的輸出層。它接受一個向量作為輸入，並將其轉換為概率分佈，其中每個元素表示一個類別的概率。
#  在多類別分類中，'softmax' 函數確保了模型的輸出是有效的概率分佈，所有概率值都介於 0 和 1 之間，並且它們的總和為 1。這使得模型能夠預測每個類別的概率，並選擇具有最高概率的類別作為預測

# input_dim 是 Keras 中 Dense 層的一個參數，它用來指定該層的輸入特徵的維度（dimension）。具體來說，它表示每個樣本的輸入特徵的數量
# input_dim 被設置為 x_train.shape[1]*x_train.shape[2]，這表示這一層的輸入特徵數量等於 x_train 的每個樣本的特徵數量。在這裡，x_train 是圖像數據經過預處理後的一維數組，
# 其特徵數量就是圖像的寬度乘以高度（x_train.shape[1] 乘以 x_train.shape[2]）。
# units 則是指定這一層的神經元數量。在這裡，設置為 800 個神經元。
# 所以，input_dim 的作用是告訴模型這一層的輸入特徵數量是多少。這是必需的，因為在深度學習中，每一層的輸入維度都需要事先指定，以確保權重矩陣的維度匹配。
# input_dim 用於指定該層的輸入特徵數量，而 units 則指定該層的神經元數量。這些參數一起定義了該層的結構和功能

# units 的數量取決於以下因素：
# 問題的複雜性：如果你處理的問題非常複雜，可能需要更多的神經元來學習複雜的模式。較大的 units 數量可以提供更多的表示能力。
# 計算資源：較大的 units 數量將需要更多的計算資源，包括記憶體和計算時間。請確保你的硬體資源足夠支援所選擇的 units 數量。
# 過度擬合（Overfitting）：如果 units 數量太多，模型可能會傾向於過度擬合訓練數據，導致在測試數據上的表現不佳。因此，你需要在訓練過程中注意監控模型的性能。
# 256 個神經元是一個合理的起始值，你可以根據模型的性能進行調整。如果模型在訓練和驗證數據上表現良好，則可以保留這個值。如果出現過度擬合或性能不佳的情況，可以嘗試減少或增加 units 的數量，
# 並通過實驗找到最適合的設置
#endregion
# mlp = Sequential()
# mlp.add(Dense(input_dim =x_train.shape[1]*x_train.shape[2], units = 512,name = 'layer_1', activation = 'relu'))
# mlp.add(Dense( units = 256,name = 'layer_2', activation = 'relu'))
# mlp.add(Dense( units = 128,name = 'layer_3', activation = 'relu'))
# mlp.add(Dense( units = 64,name = 'layer_4', activation = 'relu'))
# mlp.add(Dense( units = 3,name = 'layer_5', activation = 'softmax')) #三個類別:剪刀 石頭 布
# mlp.summary()

# 模型訓練 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
# mlp.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy']) 
# loss= 信息墒,一個衡量標準答案跟模型訓練過程腦袋的答案差距  optimizer: 用彈珠動能滾動自動找到權重最低的組合方法

# mlp 訓練
# x_train_shape：這是訓練數據的輸入特徵，通常是圖像數據經過預處理後的一維數組。它包含了訓練圖像的特徵值。
# y_train_cat：這是訓練數據的目標標籤，通常是經過 one-hot 編碼的分類標籤。它包含了每個訓練樣本的目標類別。
# batch_size：這是每次訓練時使用的小批次大小。在每個訓練時期（epoch）中，數據會被分成多個小批次來進行訓練。這個值控制每個小批次中包含多少個訓練樣本。
# epochs：這是訓練的時期數，也就是整個訓練過程將運行多少次，每一次都會使用整個訓練數據集。
# validation_split：這是用於驗證（validation）的數據集比例。通常，你會將一部分訓練數據保留下來，不參與訓練，而是用來評估模型在訓練過程中的性能。這個值表示你想要保留多少比例的數據作為驗證數據集。
# 例如，0.1 表示保留 10% 的數據作為驗證數據。
# verbose：這是訓練過程的輸出訊息級別。通常有三個設置值：
# verbose=0：不輸出訓練過程。
# verbose=1：每個時期（epoch）輸出一行訓練過程的信息。
# verbose=2：每個時期輸出一行信息，並顯示進度條
# data_history = mlp.fit(x_train_shape,
#                         y_train_cat,
#                         batch_size=100, # 隨機抽樣batch_size筆影像去訓練
#                         epochs = 80,
#                         validation_split=0.1,
#                         verbose=2)

# plt.plot(data_history.history['loss'],'ro--')
# plt.show()
# plt.plot(data_history.history['accuracy'],'bo--')
# plt.show()

# 獲取當前工作目錄的絕對路徑
current_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_dir, r'\AI_module\mlp.h5')
# str_dir = model_path[0:model_path.rfind('\\')]
# if not os.path.exists(str_dir):
#     os.makedirs(str_dir)

# 儲存模型:
# mlp.save(model_path)

time.sleep(3)

# loading 模型
model = load_model(model_path)
model.summary()

# # 驗證一下模型
# c = random.randint(0,y_train.shape[0]) # 隨機取得0 ~ y_train.shape[0]之間的數
# img = x_train[c]

# # 讀取灰階圖像
# gray_img = cv2.imread(imgs_path[c], cv2.IMREAD_GRAYSCALE)

# # 將灰階圖像轉換為 BGR 彩色格式
# bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# # 將 BGR 圖像轉換為 RGB 格式（為了 matplotlib 正確顯示）
# rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)                             

# plt.imshow(rgb_img)
# plt.title(f'img_answer:{y_train[c]}')
# plt.show()

# # 預測手勢類別
# # model.predict(x_train_shape[c:c+1,:]): 這一步使用訓練好的神經網絡模型 model 對一個圖像進行預測。 x_train_shape 是訓練數據集中所有圖像的數組，c:c+1 是選取其中一個圖像的方式。
# # 對一個圖像進行預測，並返回預測結果。結果是一個數組，每個元素代表模型對該類別的預測分數。
# # np.argmax(): 這一步使用 np.argmax() 函數找到具有最高預測分數的類別的索引。換句話說，它找到了模型認為最有可能的類別
# pred = np.argmax(model.predict(x_train_shape[c:c+1,:]))
# # 根據類別編號顯示對應的結果
# if pred == 0:
#     print(f'預測結果:剪刀')
# elif pred == 1:
#     print(f'預測結果:石頭')
# elif pred == 2:
#     print(f'預測結果:布')

answer = {'cutter':0,'stone':1,'paper':2}

def get_key_by_value(value,dictionary):
    for k,v in dictionary.items():
        if v == value:
            return k
    return None

# 這個函數使用 Pygame 库播放指定的 MP3 音頻文件。
# 它初始化 Pygame mixer，載入指定的 MP3 文件，然後播放音樂。
# 在播放音樂時，它通過 pygame.mixer.music.get_busy() 检查音樂是否仍在播放，以及使用 pygame.time.Clock().tick(10) 控制播放速度。
def play_mp3(file_name):
    pygame.mixer.init()
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

c1,c2= random.sample(list(range(x_train.shape[0])),2)
print(f'抽取兩位選手的影像位置:{c1,c2}')

plt.subplot(1,2,1) # 創建一個 1x2 的子圖網格（1 行，2 列），並選擇其中的第一個子圖（位置 1）
img1 = x_train[c1]
plt.imshow(img1,cmap = 'gray')
plt.title(f'person_1_answer:{get_key_by_value(y_train[c1],answer)}')

plt.subplot(1,2,2) # 創建一個 1x2 的子圖網格（1 行，2 列），並選擇其中的第二個子圖（位置 2）
img2 = x_train[c2]
plt.imshow(img2,cmap = 'gray')
plt.title(f'person_2_answer:{get_key_by_value(y_train[c2],answer)}')
plt.show()

print('模型預測:')
pred1 = np.argmax(model.predict(x_train_shape[c1:c1+1,:]))
pred1_str = get_key_by_value(pred1,answer)
pred2 = np.argmax(model.predict(x_train_shape[c2:c2+1,:]))
pred2_str = get_key_by_value(pred2,answer)
print(f'一號選手的模型預測:{pred1_str}')
print(f'二號選手的模型預測:{pred2_str}')

if pred1 == 0 and pred2 == 2:
    pred1=3
if pred1 == 2 and pred2 == 0:
    pred2=3

if pred2==pred1:
    print('平手')
    # play_mp3('.\AI\fair.mp3')
elif pred2>pred1:
    print('2號選手贏了')
    # play_mp3('.\AI\p2_win.mp3')
else:
    print('1號選手贏了')
    # play_mp3('.\AI\p1_win.mp3')





