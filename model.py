from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection


def train_model():
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="images")
    trainer.setTrainConfig(object_names_array=["person"], batch_size=4, num_experiments=100,
                           train_from_pretrained_model="pretrained-yolov3.h5")
    trainer.trainModel()


def detect():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("hololens-ex-60--loss-2.76.h5")
    detector.setJsonPath("detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image="holo3.jpg", output_image_path="holo3-detected.jpg")


if __name__ == "__main__":
    train_model()