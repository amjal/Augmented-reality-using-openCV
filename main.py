import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 100)
fgbg = cv2.createBackgroundSubtractorMOG2()

drop = cv2.imread('drop.png')
drop = cv2.resize(drop, (30,30))

#delete background of drop image (turn background to black)
gray = cv2.cvtColor(drop, cv2.COLOR_BGR2GRAY)

th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break
mask = np.zeros(drop.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
drop = cv2.bitwise_and(drop, drop, mask=mask)


x_offsets = np.zeros(100)
stop_sensitivity = 5
while(True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    for drop_counter in range(20):
        if x_offsets[drop_counter] == 450:
            x_offsets[drop_counter] = 0
        x_offset = x_offsets[drop_counter]
        y_offset = drop_counter*30 +10
        edge_counter = 0
        for x in range(drop.shape[0]):
            for y in range(drop.shape[1]):
                if drop[x,y,0] != 0 or drop[x,y,1] != 0 or drop[x,y,2] != 0:
                     frame[int(x_offset) +x, int(y_offset) +y] = drop[x,y]
        for y in range(drop.shape[1]):
            if fgmask[int(x_offset) + drop.shape[0]-1, int(y_offset) +y-1] != 0:
                edge_counter +=1
        if edge_counter <stop_sensitivity:
            x_offsets[drop_counter]+=2
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()