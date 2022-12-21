import cv2
import numpy as np


Def_Model = {
    'yolov3':
        {
            'weight_file': 'yolov3.weights',
            'cfg_file': 'yolov3.cfg',
            'class_file': 'coco.names'
        },
    'taxi':
        {
            'weight_file': 'yolov3_smrc_TruckDB_626videos.weights',
            'cfg_file': 'yolov3_transfer_learning_smrc.cfg',
            'class_file': 'smrc.names'
        },
    'truck':
        {
            'weight_file': 'yolov3_smrc_TruckDB_626videos.weights',
            'cfg_file': 'yolov3_transfer_learning_smrc.cfg',
            'class_file': 'smrc.names'
        },
    'face':
        {
            'weight_file': 'yolov3_smrc_sample20000_face.weights',
            'cfg_file': 'yolov3_smrc_face_or_license_plate.cfg',
            'class_file': 'class_list_face.txt'
        },
    'lp':
        {
            'weight_file': 'yolov3_smrc_sample20000_license_plate.weights',
            'cfg_file': 'yolov3_smrc_face_or_license_plate.cfg',
            'class_file': 'class_list_license_plate.txt'
        }
}


model_name = 'face'

if model_name not in Def_Model:
    model_name = 'yolov3'

weight_file = Def_Model[model_name]['weight_file']
cfg_file = Def_Model[model_name]['cfg_file']
class_file = Def_Model[model_name]['class_file']

# Load Yolo
net = cv2.dnn.readNet(weight_file, cfg_file)
classes = []
with open(class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# net, output_layers, classes, colors
def conduct_detection(
        image_path_list,
        min_score=0.05, nms_thd=0.5
    ):

    for image_path in image_path_list:
        # Loading image "0029.jpg" room_ser.jpg
        img = cv2.imread(image_path)
        # img = cv2.resize(img, None, fx=1, fy=1) #fx=0.4, fy=0.4
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > min_score:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        if len(boxes) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_score, nms_thd)

            if len(indexes) > 0:
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


test_image_path = "0029.jpg"
conduct_detection(
        image_path_list=[test_image_path],
        min_score=0.05, nms_thd=0.5
    )