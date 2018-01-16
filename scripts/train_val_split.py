import os
import shutil
from tqdm import tqdm

path_norm = '/home/bohdan/xray_project/dataset_simul/n/'
path_abnorm = '/home/bohdan/xray_project/dataset_simul/a/'



path_train_norm = '/home/bohdan/xray_project/data/train/normal'
path_train_abnorm = '/home/bohdan/xray_project/data/train/abnormal'

path_val_norm = '/home/bohdan/xray_project/data/val/normal'
path_val_abnorm = '/home/bohdan/xray_project/data/val/abnormal'


norm_imgs = list()
for (a, b, im) in os.walk(path_norm):
	norm_imgs += im

abnorm_imgs = list()

for (a, b, im) in os.walk(path_abnorm):
	abnorm_imgs += im

# moving to train and validation abnormal images
validation_percentage = 0.2
val_img_num = int(len(abnorm_imgs) * validation_percentage)
counter = 0
for ab_img in tqdm(abnorm_imgs):
	if(counter < val_img_num):
		shutil.copy(path_abnorm + ab_img,path_val_abnorm + '/')
		counter += 1
	else:
		shutil.copy(path_abnorm + ab_img,path_train_abnorm + '/' + ab_img)
print("abnormal_val: ", val_img_num)

# moving to train and validation normal images
validation_percentage = 0.2
val_img_num = int(len(norm_imgs) * validation_percentage)
counter = 0
for norm_img in tqdm(norm_imgs):
	if(counter < val_img_num):
		shutil.copy(path_norm + norm_img,path_val_norm + '/' + norm_img)
		counter += 1
	else:
		shutil.copy(path_norm + norm_img,path_train_norm + '/' + norm_img)
print("normal_val: ", val_img_num)
