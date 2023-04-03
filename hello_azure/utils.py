from skimage.feature import local_binary_pattern, hog
import cv2
import numpy as np

import os
import pickle

RawFeatureSVMModel = None
HogFeatureSVMModel = None
HistogramFeatureSVMModel = None
HarrisFeatureSVMModel = None
# ShitomasiFeatureSVMModel = None
# LBPFeatureSVMModel = None


def load_model():
  print('Loading model...')

  global RawFeatureSVMModel
  global HogFeatureSVMModel
  global HistogramFeatureSVMModel
  global HarrisFeatureSVMModel
  # global ShitomasiFeatureSVMModel
  # global LBPFeatureSVMModel

  RawFeatureSVMModel = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictionModel', 'RawSVMModel.pkl'), 'rb'))
  HogFeatureSVMModel = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictionModel', 'HogFeatureSVMModel.pkl'), 'rb'))
  HistogramFeatureSVMModel = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictionModel', 'HistogramFeatureSVMModel.pkl'), 'rb'))
  HarrisFeatureSVMModel = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictionModel', 'HarrisFeatureSVMModel.pkl'), 'rb'))
  # ShitomasiFeatureSVMModel = pickle.load(open(os.path.realpath('./hello_azure/model/ShitomasiFeatureSVMModel.pkl'), 'rb'))
  # LBPFeatureSVMModel = pickle.load(open(os.path.realpath('./hello_azure/model/LBPFeatureSVMModel.pkl'), 'rb'))

def Raw_feature(img):
   return img.ravel()

def LBP_feature(img):
  radius = 3
  n_points = 8*radius
  METHOD = 'uniform'
  fd = local_binary_pattern(img, n_points, radius, METHOD)
  return fd.ravel()

def Hog_feature(img):
  dim = (28,28)
  ppc = 6
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  fd, hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc), cells_per_block=(2,2),block_norm='L2', visualize=True)
  return fd

def Histogram_feature(img):
  fd, bins = np.histogram(img.ravel(),256,[0,256])
  return fd.ravel()

def Harris_feature(img):
  img = np.float32(img)
  dst = cv2.cornerHarris(img,2,3,0.04)
  return np.asarray(dst).ravel()

def Shitomasi_feature(img):
  corners=cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
  corners=np.int0(corners)
  fd=np.zeros((28,28))
  for i in corners: 
    x,y=i.ravel()
    fd[x][y]=1
  return fd.ravel()