# GAN : Remove image reflection

## Equipment

- NVIDIA Geforce RTX 4060 Laptop GPU 16GB

- 12th Gen Intel(R) Core(TM) i5-12500H   2.50 GHz

- RAM 16 GB 

## Dataset
SIR dataset : https://sir2data.github.io/ 

IBCLN dataset : https://github.com/JHL-HUST/IBCLN

ERRNET dataset : https://github.com/Vandermode/ERRNet

RFC dataset : https://github.com/ChenyangLEI/flash-reflection-removal?tab=readme-ov-file

I combined these three datasets and randomly selected some data to be the test datasets, while the rest were used for training.

My Classify code is Classification_image.py and Data_preprocess.py. You can refer to build the dataset for two class.

> Class1 : Reflection Class2 : NonReflection


## Model : Pix2pix

### Preface
Refer to https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix . I changed the datasets and some code.

Basically i used Pix2pix model to run this mission.I will try lots of hypothesis class to find out best accuracy.

Not only that, I'm going to try a lot of different models to compare.

### About
Use generator to generate new picture with nonreflection. And discriminator will check authenticity.

- Original(Real NonReflection Picture)
- Condition(Real Reflectuon Picture) 
- Generated(Fake NonReflection Picture)
  > producted by generator 

And we want to predict nonreflection picture from reflection picture,so the input of predict is reflection picture.(Important!)

If you input a nonreflection picture, the training process you are doing is changing the nonreflection picture into a reflection picture.

### Result
Result of combined other datasets(epoch 300 batch 8 lr 0.00005)

![image](https://github.com/user-attachments/assets/a240425d-cb0f-4733-8dd4-18e900c6e77b)
![image](https://github.com/user-attachments/assets/47df0ff9-945c-44fe-83d2-60bf3ce07630)
![image](https://github.com/user-attachments/assets/502bb303-2c5f-4887-9e63-3f6f1c0fa301)

![image](https://github.com/user-attachments/assets/e5daf8bf-54c2-401d-9775-df9cdef2e71c)
![image](https://github.com/user-attachments/assets/d2fe6f51-af9d-4670-aad9-e759a144371d)
![image](https://github.com/user-attachments/assets/a77ec678-1d91-43a5-8225-dc50246f8e00)



### Loss

This is the part of loss curve from (epoch 300 batch 8 lr 0.00005)

![image](https://github.com/user-attachments/assets/6b1e3525-6e94-4e57-a421-8cee290ca642)

![image](https://github.com/user-attachments/assets/5982fa3a-7346-4a61-a3a4-eb9093e10c38)



## Will try...
1.Location-aware Single Image Reflection Removal

>https://github.com/zdlarr/Location-aware-SIRR

2.V-DESIRR: Very Fast Deep Embedded Single Image Reflection Removal
