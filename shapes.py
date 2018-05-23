import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("00001.ppm", cv2.IMREAD_COLOR)

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




def draw_bb(img2):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img2, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)
    return img2

TARGET_DIR = "./"  # /
gt_bboxes = read_bb(TARGET_DIR + "gt.txt")
# images = read_images(TARGET_DIR)

gt = gt_bboxes['1'];  # to get only four coordinates do: gt_bboxes['0'][0][0:4]
(rows, cols) = img.shape[0:2]         # 800 by 1360

imgray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
ret,thresh = cv2.threshold(blurred,127,255, cv2.THRESH_BINARY)

edges = cv2.Canny(imgray, 100, 300)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img2 = cv2.drawContours(image, contours, -1, (255, 0,0), 3)

plt.subplot(311),
plt.imshow(img, cmap= 'brg')
plt.subplot(312),
plt.imshow(edges, cmap= 'gray')
plt.subplot(313),
plt.imshow(image, cmap= 'brg')
# plt.show()

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
img2 = img.copy();
i = 1;
for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] < 200 or M["m00"] > 3000:
        continue
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    approx = cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
    shape = str(0)
    if len(approx)==6:
        shape = "hexagon" + str(i)
        # cv2.drawContours(img2,[cnt],0,255,-1)
        img2 = draw_bb(img2)
    elif len(approx)==3:
        shape = "triangle" + str(i)
        area = cv2.contourArea(cnt)
        print ("area ", area)
        # cv2.drawContours(img2,[cnt],0,(0,255,0),-1)
        img2 = draw_bb(img2)
    elif len(approx)==4:
        shape = "square" + str(i)
        # cv2.drawContours(img2,[cnt],0,(0,0,255),-1)
        img2 = draw_bb(img2)
    elif len(approx) == 9:
        shape = "half-circle" + str(i)
        # cv2.drawContours(img2,[cnt],0,(255,255,0),-1)
    elif len(approx) > 15:
        shape = "circle" + str(i)
        img2 = draw_bb(img2)
        # cv2.drawContours(img2,[cnt],0,(0,255,255),-1)
    i = i+1


cv2.imshow('img',img2)


cv2.waitKey(0)
cv2.destroyAllWindows()
