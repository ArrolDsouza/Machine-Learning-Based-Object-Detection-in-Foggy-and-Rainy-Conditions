from ultralytics import YOLO
import cv2
import os
import torch

def test_model():
    # Print system info
    print("\nSystem Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Load the trained model
    model_path = "runs/foggy_detection/yolov8n_foggy/weights/best.pt"
    print(f"\nLooking for model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    print("Loading trained model...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Test images directory
    test_dir = "test_img1"  # Updated path for new test images
    print(f"\nLooking for test images in: {test_dir}")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        return
    
    # List test images
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"- {img}")
    
    if not test_images:
        print("No test images found!")
        return
    
    # Create output directory for results
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Process each test image
    print("\nProcessing test images...")
    for img_file in test_images:
        print(f"\nProcessing {img_file}...")
        
        # Read image
        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_file}")
            continue
        
        print(f"Image shape: {img.shape}")
        
        # Run inference
        try:
            results = model(img, conf=0.25)
            print(f"Found {len(results[0].boxes)} detections")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            continue
        
        # Draw results on image
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
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{img_file}")
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"Saved result to {output_path}")
        else:
            print(f"Error: Failed to save result to {output_path}")
    
    print("\nTesting complete! Results saved in 'test_results' directory.")

if __name__ == "__main__":
    test_model() 