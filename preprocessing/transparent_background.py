import cv2
file_name = "../data/BOBWUD_concat.png"

src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha_white = cv2.threshold(tmp, 254,255,cv2.THRESH_BINARY_INV)
_,alpha_black = cv2.threshold(tmp, 10, 255, cv2.THRESH_BINARY)
alpha = alpha_black & alpha_white
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha_black]
dst = cv2.merge(rgba,4)

trans_mask = dst[:,:,3] == 0
dst[trans_mask] = [255, 255, 255, 0]
# white_mask = dst[:,:, 0] == 255
# dst[white_mask] = [255, 0, 0, 255]

cv2.imshow('', dst)
cv2.waitKey(0)
cv2.imwrite("../data/test.png", dst)