#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:07:07 2018

@author: shehzeen
"""
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


#get_ipython().magic(u'matplotlib inline')

#get_ipython().magic(u'reload_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import pickle
from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.detector_utils import *
from lib.keras_utils import *
from lib.RandomTransform import *
from lib.OptCarlini import *
from lib.OptTransform import *
from lib.OptProjTran import *


# In[5]:

'''
# Build and load trained model
model = build_mltscl()
# model = build_cnn()
model.load_weights(WEIGTHS_PATH)

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL)

y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = keras.utils.to_categorical(y_test, NUM_LABELS)
y_val = keras.utils.to_categorical(y_val, NUM_LABELS)

# Read sign names
signnames = read_csv("./input_data/signnames.csv").values[:, 1]


# In[4]:


model.summary()


# ## Run Attacks: Adversarial Traffic Signs


'''

SAMPLE_IMG_DIR = './images/Original_Traffic_Sign_samples/'
SAMPLE_LABEL = './images/Original_Traffic_Sign_samples/labels.txt'


# Load samples 

# In[36]:


# Load sample images and labels. Remove samples
del_id = [3, 8, 9, 10, 14]
x_smp, x_smp_full, y_smp, masks, masks_full = load_samples(
    SAMPLE_IMG_DIR, SAMPLE_LABEL)

plt.ion()

'''
for im in range(len(x_smp)):
    #print(i)
    #print(signnames[predict(model, im)] + "  " + signnames[np.argmax(y_smp[i])])  
    plt.figure()
    plt.imshow(x_smp[im])
    #plt.waitforbuttonpress
    plt.show()
   # _ = raw_input("Press [enter] to continue.") # wait for input from the user
    #plt.close()    # close the figure to show the next one.


'''

'''
x_smp = np.delete(x_smp, del_id, axis=0)
x_smp_full = np.delete(x_smp_full, del_id, axis=0)
y_smp = np.delete(y_smp, del_id, axis=0)
masks = np.delete(masks, del_id, axis=0)
masks_full = np.delete(masks_full, del_id, axis=0)
'''

# Specify target classes

# In[49]:


# Target classes are chosen at random
y_target = np.array([3, 20, 42, 38,  1, 22, 24,  6, 13,  8, 29, 14, 41, 37,  7,
                     32, 19, 21,  9, 26])
y_target = keras.utils.to_categorical(y_target, NUM_LABELS)


SAVE_DIR = "./adv_signs/Adversarial_Traffic_Signs/"
alg = "opt_proj"
ind = "5"

outputs = []
norms = []
for i in range(len(y_target)):
    outputs.append(pickle.load(open("{}outputs_{}_{}_{}.p".format(
        SAVE_DIR, alg, ind, i), "rb")))
    norms.extend(pickle.load(open("{}norms_{}_{}_{}.p".format(
        SAVE_DIR, alg, ind, i), "rb")))


# Rearrange samples for evaluation function
x_adv = []
y_tg = []
y_orig = []
x_orig = []
x_orig_full = []

for i, o in enumerate(outputs):
    j = 0
    for x in o:
        if np.argmax(y_smp[i]) == np.argmax(y_target[j]):
            j += 1
        x_adv.append(x)
        y_tg.append(y_target[j])
        y_orig.append(y_smp[i])
        x_orig.append(x_smp[i])
        x_orig_full.append(x_smp_full[i])
        j += 1

x_adv = np.array(x_adv)
y_tg = np.array(y_tg)
y_orig = np.array(y_orig)
x_orig = np.array(x_orig)
x_orig_full = np.array(x_orig_full)

# Start evaluation
suc, c_adv, c_orig = evaluate_adv(model, x_adv, y_tg, x_orig, 
    y_smp=y_orig, target=True, x_smp_full=x_orig_full, tran=False)


im = x_orig[8]
print ("Original class: " + signnames[predict(model, im)])
plt.imshow(im)
plt.axis('off')
plt.show()

im = x_adv[8]
print ("Adversarial class: " + signnames[predict(model, im)])
plt.imshow(im)
plt.axis('off')
plt.show()

print(np.mean(norms))
print(suc, 'suc')
print(c_adv, 'c_adv')
print(c_orig, 'c_orig')



def find_circles(img, mg_ratio=0.4, n_circles=1):
    """
    Find circular objects and return bounding boxes in the format
    [x1, y1, x2, y2]
    """

    targetImg = np.copy(img)
    targetImg = np.uint8(targetImg * 255)
    # Apply Gaussian blur if needed
    n = 13
    targetImg = cv2.GaussianBlur(targetImg, (n, n), 0)

    # Convert to grayscale
    #grayImg = np.uint8(rgb2gray(targetImg))
    grayImg =cv2.cvtColor(targetImg,cv2.COLOR_BGR2GRAY)

    # param1 is canny threshold, param2 is circle accumulator threshold
    # Set of parameters for GTSDB testing
    # (because of different frame size and recording device)
    # circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 200,
    #                            param1=60, param2=50, minRadius=5,
    #                            maxRadius=100)
    circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0,
                               maxRadius=750)

    bboxes = []
    try:
        cir = circles.astype(np.uint16)
        for c in cir[0, :n_circles]:
            r = int(c[2])
            mg = int(r * mg_ratio)
            bboxes.append([c[0] - r - mg, c[1] - r - mg,
                           c[0] + r + mg, c[1] + r + mg])
    except AttributeError:
        pass
    except:
        raise
    return bboxes

targetImg =x_smp[9]
bbox=find_circles(targetImg, mg_ratio=0.4, n_circles=1)[0]
#targetIm2=crop_bb(targetImg, bbox)
imgtest = draw_bb(targetImg, bbox)
plt.imshow(imgtest)



