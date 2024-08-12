# GAN : Remove image reflection

## Dataset
SIR dataset : https://sir2data.github.io/ to download SIR dataset. 

IBCLN dataset : https://github.com/JHL-HUST/IBCLN

ERRNET dataset : https://github.com/Vandermode/ERRNet

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
- Condition(Real Reflectuon Picture) (input of model predict)
- Generated(Fake NonReflection Picture)
  > producted by generator 

### Result
Result of combined other datasets

![image](https://github.com/user-attachments/assets/07eb302b-077c-45b7-905e-04472376da13)

![image](https://github.com/user-attachments/assets/dfa22c0e-4641-41ea-b7d3-fe0c05d924d8)

![image](https://github.com/user-attachments/assets/1923cf97-a681-41de-bf16-68e8cf6800ea)

![image](https://github.com/user-attachments/assets/fc9189b6-ac96-4d79-b70c-fdfb85df9156)

![image](https://github.com/user-attachments/assets/47e254fd-a2d8-4e2c-b955-9440e849e781)


### Loss

This is the part of loss curve from (epoch 300 batch 8 lr 0.00005)

![image](https://github.com/user-attachments/assets/6b1e3525-6e94-4e57-a421-8cee290ca642)

![image](https://github.com/user-attachments/assets/5982fa3a-7346-4a61-a3a4-eb9093e10c38)



## Will try...
1.Location-aware Single Image Reflection Removal

>https://github.com/zdlarr/Location-aware-SIRR

2.V-DESIRR: Very Fast Deep Embedded Single Image Reflection Removal
