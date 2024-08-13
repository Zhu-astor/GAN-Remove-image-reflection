from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

class DataLoader():
    def __init__(self, img_res):
        #(128,128)
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        
        if not is_testing:
            class_type = "Train"
        else:
            class_type = "Test"
            

        path = './Dataset2/%s/Reflection/' % (class_type)
        path1 = glob('./Dataset2/%s/Reflection/*' % (class_type))
        path2 = './Dataset2/%s/NonReflection/' % (class_type)

        temp_len = len(path)

        batch_images = np.random.choice(path1, size=batch_size)

        batch_images1 = list()
        
        
        imgs_A = []
        imgs_B = []
        
        for img_path in batch_images:
            #print(img_path + "--------------reflection")
            img = self.imread(img_path)
            batch_images1.append(path2 + img_path[temp_len:])
            img = np.array(Image.fromarray(np.uint8(img)).resize(self.img_res))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
                
                
            imgs_A.append(img)

        imgs_A = np.array(imgs_A)/127.5 - 1.# normalize
            
        for img_path in batch_images1:
            #print(img_path + "--------------nonreflection")
            img = self.imread(img_path)
            
            img = np.array(Image.fromarray(np.uint8(img)).resize(self.img_res))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
            
            imgs_B.append(img)

        imgs_B = np.array(imgs_B)/127.5 - 1. # normalize

        return imgs_B, imgs_A
    
    def load_test_data(self, batch_size=1):

        path1 = glob('./Dataset2/Test/Reflection/*' )
        batch_images = np.random.choice(path1, size=batch_size)
        imgs_A = []
        
        for img_path in batch_images:
            #print(img_path + "--------------reflection")
            img = self.imread(img_path)
            img = np.array(Image.fromarray(np.uint8(img)).resize(self.img_res))
    
            imgs_A.append(img)

        imgs_A = np.array(imgs_A)/127.5 - 1.# normalize

        return imgs_A
    
    def load_batch(self, batch_size=1, is_testing=False):
        
        if not is_testing:
            class_type = "Train"
        else:
            class_type = "Test"
            

        path = './Dataset2/%s/Reflection/' % (class_type)
        path1 = glob('./Dataset2/%s/Reflection/*' % (class_type))
        path2 = './Dataset2/%s/NonReflection/' % (class_type)

        self.n_batches = int(len(path1) / batch_size)

        temp_len = len(path)
        for i in range(self.n_batches-1):
            batch = path1[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for path_A in batch:
                #print(path_A + "--------------reflection")
                path_B = path2 + path_A[temp_len:]
                #print(path_B + "--------------nonreflection")
                img_A = self.imread(path_A)
                img_B = self.imread(path_B)

                img_A = np.array(Image.fromarray(np.uint8(img_A)).resize(self.img_res))
                img_B = np.array(Image.fromarray(np.uint8(img_B)).resize(self.img_res))

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.# normalize
            imgs_B = np.array(imgs_B)/127.5 - 1.# normalize


            yield imgs_B, imgs_A


    def imread(self, path):
        return imageio.imread(path, mode='RGB').astype(np.float)
