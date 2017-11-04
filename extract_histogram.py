import cv2
import numpy as np
import csv

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

WORKING_FOLDER = '/Users/rputra/Downloads/RGB/'

ACTION = 3
SUBJECT = 8
TRIAL = 4

for a in range (0, ACTION):
    for s in range (0, SUBJECT):
        for t in range (0, TRIAL):
            print(WORKING_FOLDER + 'a' + str((a+1)) + '_' + 's' + str((s+1)) +'_t' + str((t+1)) + '_1.png')
            img = cv2.imread(WORKING_FOLDER + 'a' + str((a+1)) + '_' + 's' + str((s+1)) +'_t' + str((t+1)) + '_1.png',0)
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[100:300, 180:450] = 255
            masked_img = cv2.bitwise_and(img,img,mask = mask)
            hist_full = cv2.calcHist([img],[0],None,[256],[0,256])

            with open(WORKING_FOLDER + 'hist_full.csv', 'a') as csvfile:
                hist_writer = csv.writer(csvfile, delimiter=',')
                hist_full = np.append(hist_full,[a+1])
                hist_writer.writerow(hist_full)

