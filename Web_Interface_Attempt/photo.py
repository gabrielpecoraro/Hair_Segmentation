from picamera import PiCamera
from time import sleep

camera = PiCamera()

# Capture une photo
camera.capture('/home/pi/Documents/photo/test_image.jpg')

# Capture une vidéo de 5 secondes
camera.start_recording('/home/pi/Documents/photo/test_video.h264')
sleep(5)
camera.stop_recording()

