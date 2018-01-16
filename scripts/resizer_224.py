#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
from tqdm import tqdm

# RESIZING
def readf(new_img_size, source_folder, output_folder):
    try:
        input_dir  = source_folder.rstrip('/')  # path to img source folder
        img_size   = str(new_img_size)  # The image size (128, 256,etc)
        output_dir  = output_folder.rstrip('/') # output directory
        print("starting....")
        print ("Colecting data from %s " % input_dir)
        tclass = [ d for d in os.listdir( input_dir ) ]
        counter = 0
        strdc = ''
        hasil = []
        for x in tqdm(tclass):
            list_dir =  os.path.join(input_dir, x )
            list_tuj = os.path.join(output_dir+'/', x+'/')
#             print()
            if not os.path.exists(list_tuj):
                os.makedirs(list_tuj)
            if os.path.exists(list_tuj):
                for d in os.listdir(list_dir):
#                     try:
                    img = Image.open(os.path.join(input_dir+'/'+x,d))
                    img = img.resize((int(img_size),int(img_size)),Image.ANTIALIAS)
                    fname,extension = os.path.splitext(d)
                    newfile = fname+extension
                    if extension != ".png" :
                        newfile = fname + ".png"
                    img.save(os.path.join(output_dir+'/'+x,newfile),"PNG",quality=90)
#                     print ("Resizing file : %s - %s " % (x,d))
#                     except Exception,e:
#                         print ("Error resize file : %s - %s " % (x,d))
#                         sys.exit(1)
                counter +=1
    except Exception,e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
readf(224, '/gpfs/rocket/dmytro/xray/dataset', '/gpfs/rocket/dmytro/xray/dataset_resized_224')
