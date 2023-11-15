# UNI
# import cv2

# captura = cv2.VideoCapture(0)
# auxiliar = 0
# while True:

#     ret, frame = captura.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     if ret == False:
#         break

#     if auxiliar == 40:
#         fondo = gray

#     if auxiliar > 40:
#         diferencia = cv2.absdiff(gray, fondo)
#         #cv2.imshow('diferentes', diferencia)
#         _, th = cv2.threshold(diferencia, 40, 255, cv2.THRESH_BINARY)
#         #cv2.imshow('th', th)
#         imagen,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for i in imagen:
#             area = cv2.contourArea(i)
#             if area > 9000:
#                 x,w,y,z = cv2.boundingRect(i)
#                 cv2.rectangle(frame, (x,y), (x+w, y+z), (0,0,255), 3)
#     auxiliar += 1


#     cv2.imshow('Captura', frame)
#     #cv2.imshow('GRISES', gray)
#     valor = cv2.waitKey(1)

#     if valor == 27:
#         break


# captura.release()
# cv2.destroyAllWindows()

import cv2

face_cacade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cacade.detectMultiScale(gray, 1.1, 4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('img',  img)
        k = cv2.waitKey(30)
        if k == 27:
            break
        cap.release() 


