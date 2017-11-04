import scipy.io
import csv
import numpy as np

def import_depth_data(action, subject, trial):
    filename = 'a'+str(action)+'_s'+str(subject)+'_t'+str(trial)+'_depth.mat'
    mat = scipy.io.loadmat(filename)
    ###depthData=[]
    
    print len(mat['d_depth'])
   
    #for a in range(0, len(mat['d_depth'])):
    #    print len(mat['d_depth'][a])
    #    for b in range(0, len(mat['d_depth'][a])):
    #        print len(mat['d_depth'][a][b])
    #        for c in range(0, len(mat['d_depth'][a][b])):
    #            with open('a24_s8_t3_depth.csv', 'a') as csvfile:
    #                writer = csv.writer(csvfile, delimiter=',')
                    ###depthData = np.append(depthData,mat['d_depth'][a][b][c])
    #                writer.writerow(mat['d_depth'][a][b])
    for c in range(0, len(mat['d_depth'])):
        with open('a24_s8_t3_depth.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(mat['d_depth'][:][:][c])
    
    return mat['d_depth']
    
print import_depth_data(27,8,3)
