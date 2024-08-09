# GAN : Remove image reflection

## Dataset
The file is too big . I can't put on my github . You can go to https://sir2data.github.io/ to download SIR dataset. 

My Classify code is Classification_image.py and Data_preprocess.py. You can refer to build the dataset for two class.

> Class1 : Reflection Class2 : NonReflection


## Model : Pix2pix

### Preface
Refer to https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix . I changed the datasets and some code.

Basically i used Pix2pix model to run this mission.I will try more model for comparing accuracy later.

### Result
If you use my code.You may get some result like this.

The layer and hypothesis class can fine tune.

![image](https://github.com/user-attachments/assets/4bdbe444-b6da-4a29-b4e1-c32342dbe5e8)

![image](https://github.com/user-attachments/assets/eb4d779a-9c23-4ae2-a3fd-e87c770e3fbd)

![image](https://github.com/user-attachments/assets/eb25d601-fc5a-4788-9c98-959898e4134e)

![image](https://github.com/user-attachments/assets/53253b46-6a4b-490b-b093-bba05c9be54f)

### Loss

I will update loss value later...

![image](https://github.com/user-attachments/assets/9c41ef90-5030-4f74-997a-e7eeb4238629)

## Will try...
1.Location-aware Single Image Reflection Removal

>https://github.com/zdlarr/Location-aware-SIRR

2.V-DESIRR: Very Fast Deep Embedded Single Image Reflection Removal

This is the last epoch's information.



