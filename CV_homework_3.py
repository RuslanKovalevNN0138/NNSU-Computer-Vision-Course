from cmath import pi
import math
import numpy as np
import statistics as st
import cv2 as cv

balls = cv.imread("balls.jpg")
grey = cv.cvtColor(balls, cv.COLOR_BGR2GRAY)
a, image = cv.threshold(grey, 127, 255, 0)

k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

image = cv.morphologyEx(image, cv.MORPH_OPEN, k1, iterations=2)
image = cv.morphologyEx(image, cv.MORPH_CLOSE, k1, iterations=5)
image = cv.erode(image, k2, iterations=7)
image = cv.dilate(image, k2, iterations=4)

contours, b = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(balls, contours, -1, (0, 255, 0), 3)
radius =[]
for i in contours:
    radius.append(cv.minEnclosingCircle(i)[1])

print("количество шаров: ", len(radius), "\nсредний радиус: ", st.mean(radius), "\nдисперсия: ", np.var(radius))

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.imshow('image', balls)

cv.waitKey()
cv.destroyAllWindows()