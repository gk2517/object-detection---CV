import cv2
import winsound

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gaussian_blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 3000:
            continue
        x, y, l, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + l, y + h), (0, 255, 0), 2)
        winsound.Beep(1000, 100)
    
    cv2.imshow('Motion Cam', frame1)
    if cv2.waitKey(10) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
