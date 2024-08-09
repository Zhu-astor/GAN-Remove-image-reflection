from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
    
img_res = (256,256)    
    
class_type = "Test"    


# data_loader
# path = './Dataset/%s/Reflection/' % (class_type)
# path1 = glob('./Dataset/%s/Reflection/*' % (class_type))
# path2 = './Dataset/%s/NonReflection/' % (class_type)

# temp_len = len(path)

# batch_images = np.random.choice(path1, size=1)

# batch_images1 = list()
# batch_images1.append(path2 + batch_images[-1][temp_len:])

# print(batch_images)
# print(batch_images1)

# imgs_A = []
# imgs_B = []
        
# for img_path in batch_images:
#     img = imageio.imread(img_path, mode='RGB').astype(np.float)
      
#     img = np.array(Image.fromarray(np.uint8(img)).resize(img_res))

#     imgs_A.append(img)

# imgs_A = np.array(imgs_A)/127.5 - 1.
    
# for img_path in batch_images1:
#     img = imageio.imread(img_path, mode='RGB').astype(np.float)
      
#     img = np.array(Image.fromarray(np.uint8(img)).resize(img_res))

#     imgs_B.append(img)

# imgs_B = np.array(imgs_B)/127.5 - 1.

# print(imgs_A)
# print("----")
# print(imgs_B)
    
    
batch_size = 1
            
path = './Dataset/%s/Reflection/' % (class_type)
path1 = glob('./Dataset/%s/Reflection/*' % (class_type))
path2 = './Dataset/%s/NonReflection/' % (class_type)

n_batches = int(len(path1) / batch_size)

temp_len = len(path)

for i in range(n_batches-1):
    batch = path1[i*batch_size:(i+1)*batch_size]
    imgs_A, imgs_B = [], []
    for path_A in batch:

        path_B = path2 + path_A[temp_len:]
        img_A = imageio.imread(path_A, mode='RGB').astype(np.float)
        img_B = imageio.imread(path_B, mode='RGB').astype(np.float)

        img_A = np.array(Image.fromarray(np.uint8(img_A)).resize(img_res))
        img_B = np.array(Image.fromarray(np.uint8(img_B)).resize(img_res))

        imgs_A.append(img_A)
        imgs_B.append(img_B)

    imgs_A = np.array(imgs_A)/127.5 - 1.
    imgs_B = np.array(imgs_B)/127.5 - 1.
    
print(imgs_A)
print(imgs_B)





    



