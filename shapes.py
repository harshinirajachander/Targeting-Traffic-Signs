import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("00000.ppm", cv2.IMREAD_COLOR)

def read_bb(filename):
    """Read ground truth bounding boxes from file"""

    gt_bboxes = {}
    with open(filename) as file:
        for line in file:
            frame = str(int(line.strip().split(".")[0]))
            bb = line.strip().split(";")[1:6]
            bb = list(map(int, bb))
            if frame in gt_bboxes:
                gt_bboxes[frame].append(bb)
            else:  
                gt_bboxes[frame] = [bb]
    return gt_bboxes

def draw_bb(img, bbox, color=(0, 1, 0)):
    """Draw bounding box"""
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color, 2)
    return img


TARGET_DIR = "./"  # /
gt_bboxes = read_bb(TARGET_DIR + "gt.txt")
# images = read_images(TARGET_DIR)

gt = gt_bboxes['0'];  # to get only four coordinates do: gt_bboxes['0'][0][0:4]
(rows, cols) = img.shape[0:2]         # 800 by 1360

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
edges = cv2.Canny(imgray, 70, 250)
image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img2 = cv2.drawContours(image, contours, 1, (255, 0,0), 3)

plt.subplot(311),
plt.imshow(img, cmap= 'brg')
plt.subplot(312),
plt.imshow(edges, cmap= 'gray')
plt.subplot(313),
plt.imshow(img2, cmap= 'brg')


# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)
# # plt.show()
# cv2.imwrite('houghlines3.jpg', img)

cv2.imshow('hough', imgray)


cv2.waitKey(0)
cv2.destroyAllWindows()