import shutil
import os
from shutil import copy


#--------------------------------------------------------------------------------------------------------------
#file_path = "C:/Users/bubbl/Downloads/SIR2/Wildscene"
#for i in range(1,109):
#    file_exist = False
#    f1 = file_path + "/" + (str)(i) + "/m.jpg"
#    if os.path.isfile(f1):
#        print("檔案存在")
#        file_exist = True
#    else:
#        print("檔案不存在")
#        f1 = file_path + "/" + (str)(i) + "s/m.jpg"
#        if os.path.isfile(f1):
#            file_exist = True
#    if file_exist == True:
#        f2 = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/Dataset/reflection/image_" + (str)(i) + ".jpg"
#        shutil.copyfile(f1,f2)
#--------------------------------------------------------------------------------------------------------------
#file_path = "C:/Users/bubbl/Downloads/SIR2/SolidObjectDataset/SolidObjectDataset"
#num = 0
#for i in range(1,20):
#    for j in range (1,32):
#        file_exist = False
#        f1 = file_path + "/" + (str)(i) + "/Focus/" + (str)(j)  + "/m.jpg"
#        if os.path.isfile(f1):
#            print("檔案存在")
#            file_exist = True
#            num = num + 1
#        else:
#            print("檔案不存在")
#        if file_exist == True:
#            f2 = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/Dataset/reflection/image_" + (str)(num+147) + ".jpg"
#            shutil.copyfile(f1,f2)
#--------------------------------------------------------------------------------------------------------------

folder_name = ["ab","ae","ah","ai","ba","be","bh","bi","ea","eb","eh","ei","ha","hb","he","hi","ia","ib","ie","ih"]
file_path = "C:/Users/bubbl/Downloads/SIR2/Postcard Dataset/Postcard Dataset/Focus"
num = 0
for i in folder_name:
    for j in range(1,32):
        file_exist = False
        f1 = file_path + "/" + i + "/" + (str)(j)  + "/" + i +"-5-"+"m-"+str(j)+".png"
        if os.path.isfile(f1):
            print("檔案存在")
            file_exist = True
            num = num + 1
        else:
            print("檔案不存在")
        if file_exist == True:
            f2 = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/Dataset/reflection/image_" + (str)(num+300) + ".jpg"
            shutil.copyfile(f1,f2)