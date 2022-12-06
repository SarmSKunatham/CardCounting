import cv2

im = cv2.imread('/Users/sarmkunatham/Documents/main/github/CardCounting/cardsdataset/test/test1.jpeg')[..., ::-1]
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

cv2.imshow('Original', im)

gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
cv2.imshow('Gray', gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blur', blur)

_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Otsu thresholded', thresh)

cv2.waitKey(0)