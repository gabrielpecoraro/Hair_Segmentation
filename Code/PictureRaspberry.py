import cv2
import time

def capture_photo(filename):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    # Capture a single frame
    ret, frame = cap.read()
    if ret:
        # Save the image
        cv2.imwrite(filename + '.jpg', frame)
        print(f"Photo saved as {filename}.jpg")
    else:
        print("Error: Unable to capture photo.")
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Example usage
capture_photo("captured_image")
