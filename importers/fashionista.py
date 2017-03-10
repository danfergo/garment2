import PIL

import cv2
import numpy
import scipy.io


DATASET_PATH = '../../../Datasets/fashionista-v0.2.1/'
MAT_FILE = DATASET_PATH + 'fashionista_v0.2.1.mat'

mat = scipy.io.loadmat(MAT_FILE, squeeze_me=True)
# print type(mat.get('truths')[0])


print '__globals__'
print mat['__globals__']
print '\n'

print '__header__'
print mat['__header__']
print '\n'

print '__version__'
print mat['__version__']
print '\n'

# print type(mat['truths'][0]['image'])
img = PIL.Image.fromarray(mat['truths'][0]['image'])
# print numpy.uint8(mat['truths'][0]['image']).shape
x = numpy.asarray(img)
print x.shape
# cv2.imshow("image", x)
# cv2.waitKey(0)

# for m in mat:
#     print type(mat[m])
#     print '\t' + mfor m in mat:
#     print type(mat[m])
#     print '\t' + m
