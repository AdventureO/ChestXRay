import os
import shutil
from tqdm import tqdm
path_to_dir = '/gpfs/rocket/dmytro/xray/dataset_resized_64/'
dest = '/gpfs/rocket/dmytro/xray/all'
all_folders = [x[0] for x in os.walk(path_to_dir)]

for folder in tqdm(all_folders):
	if(folder == path_to_dir):
		continue
	all_img = [x[2] for x in os.walk(folder)][0]
	for img in all_img:
		src = folder + '/' + img
		#print(src)
		shutil.copy(src, dest)
		
		


