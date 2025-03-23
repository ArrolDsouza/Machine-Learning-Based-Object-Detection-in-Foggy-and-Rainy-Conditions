from ultralytics import YOLO
import torch
import os

def main():
    # Print system info
    print("\nSystem Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

    # Print current working directory
    print(f"\nCurrent working directory: {os.getcwd()}")

    # Check if model file exists
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"\nError: Model file {model_path} not found!")
        return

    # Load the model
    print("\nLoading YOLOv8n model...")
    model = YOLO(model_path)

    # Check if dataset.yaml exists
    yaml_path = "data/foggy_cityscapes/dataset.yaml"
    if not os.path.exists(yaml_path):
        print(f"\nError: Dataset configuration file {yaml_path} not found!")
        return

    # Print dataset information
    print("\nDataset Configuration:")
    print(f"- Dataset YAML: {yaml_path}")
    print("- Training images: data/foggy_cityscapes/train/images")
    print("- Validation images: data/foggy_cityscapes/val/images")
    print("- Number of classes: 8")

    # Check if training images exist
    train_dir = "data/foggy_cityscapes/train/images"
    if not os.path.exists(train_dir):
        print(f"\nError: Training directory {train_dir} not found!")
        return
    train_images = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"- Number of training images: {train_images}")

    # Check if validation images exist
    val_dir = "data/foggy_cityscapes/val/images"
    if not os.path.exists(val_dir):
        print(f"\nError: Validation directory {val_dir} not found!")
        return
    val_images = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"- Number of validation images: {val_images}")

    # Start training
    print("\nStarting training...")
    try:
        results = model.train(
            data=yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            name="yolov8n_foggy",
            project="runs/foggy_detection",
            exist_ok=True,
            verbose=True
        )
        print("\nTraining completed successfully!")
        print(f"Results saved in: {results}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 