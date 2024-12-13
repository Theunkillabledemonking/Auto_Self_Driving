# -*- coding: utf-8 -*-
import cv2

def main():
    # Connect to the webcam (device index is usually 0)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Cannot open webcam.")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print("Cannot retrieve frame.")
            break
        
        # Display the frame
        cv2.imshow('Webcam Video', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
