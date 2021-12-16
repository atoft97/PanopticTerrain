import cv2

imgPath = "latex/plainsDrive_frame01952Semantic.png"

originalImage = cv2.imread(imgPath)
newImge = (originalImage +5) *10
print(newImge)

cv2.imwrite("latex/plainsDrive_frame01952Semantic2.png", newImge)