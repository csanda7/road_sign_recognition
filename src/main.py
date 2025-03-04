import cv2

image = cv2.imread('data/speed_limit_50.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()