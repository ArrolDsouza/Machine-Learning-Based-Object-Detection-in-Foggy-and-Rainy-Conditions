from ultralytics import YOLO
import cv2
import torch
import time
import os

def run_camera_detection():
    # Print system info
    print("\nSystem Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Load the trained model
    model_path = "runs/foggy_detection/yolov8n_foggy/weights/best.pt"
    print(f"\nLoading model from: {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create output directory for saving frames
    output_dir = "camera_feed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize FPS counter
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("\nStarting camera feed. Press Ctrl+C to stop.")
    print("Frames will be saved to the 'camera_feed' directory.")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame!")
                break
            
            # Run inference
            results = model(frame, conf=0.25)
            
            # Draw results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:  # Update FPS every 30 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames. FPS: {fps:.1f}")
            
    except KeyboardInterrupt:
        print("\nStopping camera feed...")
    finally:
        # Clean up
        cap.release()
        print("Camera feed stopped.")

if __name__ == "__main__":
    run_camera_detection() 