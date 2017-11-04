import numpy as np
import os
import glob
import math
import cv2
import random
import cPickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import svm
import time
from PIL import Image
from scipy import linalg as LA
import fnmatch

class HOGCompute:
    def __init__(self):

        FilePath = 'video_data/Training/'
        Folders = os.listdir(FilePath)
        # e.g. Folders = a1, a2, a3
        LabelCount = 1 # 3 classes for classification e.g. swipe left, right and wave
        #FolderCheck = False
        Labels = np.array([])

        output = self.GetTestData()

        #FPA = np.zeros((48,2))
        #FPACount = 0
        FPA = np.array([])

        for FileName in Folders:
            if FileName.startswith('.'):
                continue
            pathToFolder = FilePath + FileName + "/"
            # e.g. pathToFolder = video_data/Training/a1/
            newEntry = os.listdir(pathToFolder)
            # e.g. newEntry = a1_s1/,...
            print "Label is: ", LabelCount
        
            for VideoEntry in newEntry:
                if VideoEntry.startswith('.'):
                    continue
                pathToVideoFile = pathToFolder + VideoEntry
                # e.g. pathToVideoFile = video_data/Training/a1/a1_s1
                images = self.sort_files(pathToVideoFile)
                print("Training images : " + str(images))
                Number_of_Frames = len(images)
                print("Number_of_Frames : " + str(Number_of_Frames))
                height = 480
                width = 640

                samples = np.array([[0 for x in range(0,Number_of_Frames)] for x in range(0,(height*width))])
                BigCount = 1
                sampleIndex = 0

                for i in range(0,Number_of_Frames):
                    imgPath = pathToVideoFile + "/" + images[i] + ".png"
                    print("training data : " + imgPath)
                    try:
                        #Read the image
                        img = cv2.imread(imgPath,0)
                        img = cv2.resize(img, (height, width))

                        #Vectorize the image
                        tempArray = np.reshape(img,(height*width)).T

                        #add to the big array
                        samples[:,sampleIndex]= np.copy(tempArray)
                        sampleIndex+=1
                    except:
                        print("image not found...skipping")
                        continue
                    
                Labels = np.append(Labels, LabelCount)
                data, eigenValues, eigenVectors = self.PCA(samples)
                print('training data PCA completed : ' + str(eigenVectors))

                input = eigenVectors
                print('input.shape : ' + str(input.shape))
                print('output.shape : ' + str(output.shape))
                print('min.input : ' + str(np.argmin(input)))
                print('min.output : ' + str(np.argmin(output)))

                mat = np.dot(input.T, output)

                U, ss, Vh = LA.svd(mat, full_matrices= False)

                angles = np.array([np.arccos(e) for e in ss])
                print "Principal angles: ", angles

                FPA = np.append(FPA, angles)
                #FPA[FPACount,:] = angles
                #FPACount += 1

                #PA = self.angle_between(v1,v2)
                #dotProduct = sum((a*b) for a, b in zip(v1, v2))
                print FPA.shape
            LabelCount += 1

        print(FPA)
        indx = np.argmin(FPA)
        print('Labels : ' + str(Labels))
        print(indx)
        
        print self.DisplayAction(Labels[indx])

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def GetTestData(self):
        TEST_FILE = 'a1_s1_t'
        #TEST_FILE = 'a3_s7_t'
        #TEST_FILE = 'a2_s8_t'
        #TEST_FILE = 'a1_s8_t'
        
        FilePath = 'video_data/Test/'
        #Folders = os.listdir(FilePath)
        print("preparing test data")
        #for FileName in Folders:
        #images = self.sort_files(FilePath + FileName)
        #images = self.sort_files(FilePath, TEST_FILE + '*.png')
        #Number_of_Frames = len(images)
        #print("test images array : " + str(images) + " : " + str(Number_of_Frames))
        height = 480
        width = 640
        Number_of_Frames = 32
        Testsamples = np.array([[0 for x in range(0,Number_of_Frames)] for x in range(0,(height*width))])
        BigCount = 1
        sampleIndex = 0
        
        for t in range(1,5):
            numOfFiles=len(fnmatch.filter(os.listdir(FilePath),TEST_FILE + str(t) + '_*.png'))
            print('t : ' + str(t) + ', numOfFiles : ' + str(numOfFiles))
            for i in range(1,(numOfFiles+1)):
                try:
                    #TestimgPath = FilePath + str(FileName) + "/0 (" + str(i) + ").jpg"
                    TestimgPath = FilePath + TEST_FILE + str(t) + "_" + str(i) + ".png"
                    print("test data : " + TestimgPath)
                    #Read the image
                    img = cv2.imread(TestimgPath,0)
                    img = cv2.resize(img, (height, width))

                    #Vectorize the image
                    tempArray = np.reshape(img,(height*width)).T

                    #add to the big array
                    Testsamples[:,sampleIndex]= np.copy(tempArray)
                    sampleIndex+=1
                except:
                    print('Error occured..continue loop')
        print('test data prepared, running PCA on test data...')
        data, eigenValues, eigenVectors = self.PCA(Testsamples)
        print('test data PCA completed : ' + str(eigenVectors))
        return eigenVectors
        
    def PCA(self,data):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        data = np.float64(data)
        
        dims_rescaled_data=2
        m, n = data.shape

        data -= data.mean(axis=0)

        # calculate the covariance matrix
        R = np.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = LA.eig(R)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs

    def DisplayAction(self,actionIndex):
        if(actionIndex == 1):
            Action = "Swipe left"
        elif(actionIndex== 2):
            Action = "Swipe right"
        elif(actionIndex== 3):
            Action = "Wave"
        #elif(actionIndex == 4):
        #    Action = "Running"
        #elif(actionIndex == 5):
        #    Action = "Walking"
        return Action

    def sort_files(self, index):
        self.fname=[]
        path = str(index) + "/*.*"
        for file in sorted(glob.glob(path)):
            #print(file)
            s=file.split ('/')
            a=s[-1].split('.')[0]
            #x=a[-1].split('.')
            #o= x[0].split('(')[1]
            #o = o.split(')')[0]
            #self.fname.append(int(o))
            self.fname.append(a)
        return(sorted(self.fname))

if __name__=='__main__':
    start_time = time.time()
    h= HOGCompute()
    print("--- %s seconds ---" % (time.time() - start_time))