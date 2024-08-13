import glob
import os
images = glob.glob('C:/Users/bubbl/Desktop/Contest/AI GO/github/Dataset2/Test/Reflection/*')
n = 0
for i in images:
    os.rename(i,f'C:/Users/bubbl/Desktop/Contest/AI GO/github/Dataset2/Test/Reflection/{n:03d}.jpg')
    n = n + 1