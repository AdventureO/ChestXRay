import pandas as pd 
import os
import shutil
from tqdm import tqdm



normal = '/home/bohdan/xray_project/dataset_simul/n'
abnormal = '/home/bohdan/xray_project/dataset_simul/a'
current_folder = '/home/bohdan/xray_project/all/'
files_dict = dict()
files = list()
for (a, b, im) in os.walk(current_folder):
	files += im
with open('images_labels.txt', 'r') as f:
	for line in f:
		name_label = line.split()
		files_dict[name_label[0]] = int(name_label[1])

for img in files:
	src = current_folder + img
	if(files_dict[img] == 1):
		shutil.move(src,  abnormal + '/' + img)
	else:
		shutil.move(src, normal + '/' + img)


