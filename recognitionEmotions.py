import cv2
import os
import numpy as np
import smtplib
import time

def emotionImage(emotion):
	# Emojis
	if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
	if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
	if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
	if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
	if emotion == 'Silencio': image = cv2.imread('Emojis/silencio.jpeg')
	if emotion == 'Sueldo': image = cv2.imread('Emojis/sueldo.jpeg')
	return image

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.createLBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')
# --------------------------------------------------------------------------------

dataPath = '/facialRecognition/Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

emotion_times = {emotion: 0 for emotion in imagePaths}
total_time = 0

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

start_time = time.time()  # Iniciar el contador de tiempo

while True:

	ret,frame = cap.read()
	if ret == False: break
	 	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
  
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = emotion_recognizer.predict(rostro)
		
		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		
		elapsed_time = time.time() - start_time  # Calcula el tiempo transcurrido
		total_time += elapsed_time  # Suma el tiempo al total
		
		if result[1] < umbral:
			emotion_detected = imagePaths[result[0]]
			emotion_times[emotion_detected] += elapsed_time
  
		# EigenFaces
		if method == 'EigenFaces':
			if result[1] < 5700:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# FisherFace
		if method == 'FisherFaces':
			if result[1] < 500:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# LBPHFace
		if method == 'LBPH':
			if result[1] < 60:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

	cv2.imshow('nFrame',nFrame)
	k = cv2.waitKey(1)
	if k == 27:
		break

# Calcular la emoción que estuvo presente por más tiempo y su porcentaje de tiempo
most_present_emotion = max(emotion_times, key=emotion_times.get)
percentage_time = (emotion_times[most_present_emotion] / total_time) * 100

print(f"La emoción más presente fue '{most_present_emotion}' durante {emotion_times[most_present_emotion]} segundos.")
print(f"Representa aproximadamente el {percentage_time:.2f}% del tiempo total.")# Calcular la emoción que estuvo presente por más tiempo y su porcentaje de tiempo

# Enviar correo con la información recopilada
def send_email(emotion, time_detected, most_present_emotion, percentage_time):
    
    gmail_user = '@gmail.com' 
    gmail_password = '' 
    sent_from = gmail_user
    to = ['']  
    subject = 'Resumen de detección de emociones'
    body = (f"Emoción detectada durante el tiempo: '{emotion}' ({time_detected} segundos).\n\n"
            f"La emoción más presente fue '{most_present_emotion}' durante {emotion_times[most_present_emotion]} segundos, "
            f"representando aproximadamente el {percentage_time:.2f}% del tiempo total.")

    email_text = f"""\
    From: {sent_from}
    To: {", ".join(to)}
    Subject: {subject}

    {body}
    """

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
        print('Correo electrónico enviado exitosamente.')
    except Exception as e:
        print(f'Error al enviar el correo electrónico: {e}')

send_email(emotion_detected, emotion_times[emotion_detected], most_present_emotion, percentage_time)

cap.release()
cv2.destroyAllWindows()
