from imageai.Detection.Custom import CustomObjectDetection
# import os

# execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
# detector.setModelPath( os.path.join(execution_path , "firstSense/mytest/hololens-yolo/models/yolov3_hololens-yolo_last.pt"))
detector.setModelPath("firstSense/mytest/hololens-yolo/models/yolov3_hololens-yolo_last.pt")
detector.setJsonPath("firstSense/mytest/hololens-yolo/json/hololens-yolo_yolov3_detection_config.json")
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "firstSense/mytest/hololens-yolo/train/images/image (1).jpg"), output_image_path=os.path.join(execution_path , "firstSense/mytest/holo+.jpg"), minimum_percentage_probability=30)
detections = detector.detectObjectsFromImage(input_image="firstSense/mytest/hololens-yolo/train/images/image (231).jpg", output_image_path="firstSense/mytest/holo+.jpg", minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")