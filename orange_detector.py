import numpy as np
import cv2
webcam=cv2.VideoCapture(0)
while(1):
    _, imageFrame=webcam.read()
    hsvFrame=cv2.cvtColor(imageFrame,cv2.COLOR_BGR2HSV)

    orange_lower=np.array([10,100,100],np.uint8)
    orange_upper=np.array([25,255,255],np.uint8)
    orange_mask=cv2.inRange(hsvFrame,orange_lower,orange_upper)

    kernel =np.ones((5,5),"uint8")

    orange_mask=cv2.dilate(orange_mask,kernel)

    Filtered_frame=cv2.bitwise_and(imageFrame,imageFrame,mask=orange_mask)

    contours, heirarchy = cv2.findContours(orange_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for  pic, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if area>300:
            x,y,w,h=cv2.boundingRect(contour)
            imageFrame=cv2.rectangle(imageFrame, (x,y),(x+w,y+h),(0,130,255),2)
            cv2.putText(imageFrame,"ORANGE",(x,y), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,130,255))
    cv2.imshow("Original picture",imageFrame)
    cv2.imshow("Orange",Filtered_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break

