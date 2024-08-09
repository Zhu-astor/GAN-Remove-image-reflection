import os
import numpy as np
from PIL import Image

# 設定圖片尺寸
IMG_HEIGHT = 128
IMG_WIDTH = 128

# 載入圖片的函數
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB') # 確保圖片是 RGB 格式
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

# 載入反光和非反光圖片
reflective_images, reflective_labels = load_images_from_folder('C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/Dataset/reflection', 1)
non_reflective_images, non_reflective_labels = load_images_from_folder('C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/Dataset/non_reflection', 0)

# 合併數據
images = np.array(reflective_images + non_reflective_images)
labels = np.array(reflective_labels + non_reflective_labels)

# 打亂數據
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# 正規化圖片數據
images = images / 255.0

print(f"Loaded {len(images)} images.")

# 保存數據
np.save('images.npy', images)
np.save('labels.npy', labels)
