import cv2


cam = cv2.VideoCapture(0)

result, image = cam.read()
if result:
    
    # show the image
    cv2.imshow("test", image)
    
    # save the image
    cv2.imwrite("test1.png", image)