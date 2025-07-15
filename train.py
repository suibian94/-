from ultralytics import YOLO

def main():
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("./model/yolov8n.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="aa.yml",
        epochs=10,
        imgsz=320,
        batch=200,  # 降低批次大小（如2或4，根据内存调整）
        workers=0)  # Windows必须设为0（禁用多进程）)

    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    results = model("../yolo_image/images/fimg_3.jpg")
    return results
if __name__ == '__main__':
    aa=main()
    print(aa)