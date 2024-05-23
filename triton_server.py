import cv2
import traceback
import numpy as np
import tritonclient.grpc as grpcclient

class YoloV5_Triton_Client:
    def __init__(
    self,
    model_name = "yolov5",
    detection_confidence=0.5,
    grpc_host="localhost",
    grpc_port=8001,
    nms_thresh = 0.6,       
    force_float_input=True,
    filter_class_id = None,
    input_label = "data",
    output_label = "prob",
    ):
        self.force_float_input = force_float_input
        self.grpc_port      = grpc_port
        self.grpc_host      = grpc_host
        self.model_name     = model_name
        self.input_width    = 640
        self.input_height   = 640
        self.conf_thresh    = detection_confidence
        self.nms_thresh     = nms_thresh
        self.filter_class_id= filter_class_id
        self.input_label    = input_label
        self.output_label   = output_label

        self.inputs  = []
        self.outputs = []

        if self.force_float_input:
            dtype = "FP32"
        else:
            dtype = "UINT8"

        self.inputs.append(grpcclient.InferInput(
                self.input_label,
                [1, 3, self.input_width, self.input_height],
                dtype))

        self.outputs.append(
            grpcclient.InferRequestedOutput(self.output_label))

        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=f"{self.grpc_host}:{self.grpc_port}")

        except Exception as e:
            logging.error(
                f"{__module_name__} Channel Creation Failed: {e}")
            raise Exception(
                f"{__module_name__} Channel Creation Failed: {e}")
        
        # Check Model Architecture (default model is yolov5)
        model_arch = self.model_name.lower().split("_")[0][:6]
        if model_arch == "yolov8":
            self.model_yolov8 = True
            self.filter_class_id = None
        else :
            self.model_yolov8 = False

        # self.model_yolov8 = False
    def infer(self, input_image_path):
        try:
            input_image, origin_h, origin_w = self.preprocess(
                input_image_path
            )
            # Initialize the data
            self.inputs[0].set_data_from_numpy(input_image)
            # Test with outputs
            results = self.triton_client.infer(
                model_name=self.model_name, inputs=self.inputs, outputs=self.outputs,headers={'test': '1'}
            )
            # Get the output arrays from the results
            if self.model_yolov8:
                output = np.transpose(np.squeeze(results.as_numpy(self.output_label))) #.transpose((0, 2, 1))[0, :, :]
            else: 
                output = results.as_numpy(self.output_label)[0, :, :]  #results.as_numpy(self.output_label)[0,:,:]
            # print(output)
            # Do postprocess
            result_boxes = self.post_process(output, origin_h, origin_w)

            return result_boxes

        except Exception as e:
            logging.error(f"{__module_name__} Inference Error: {e}")
            logging.error(f"{__module_name__}:\n {traceback.format_exc()}")
    
    def preprocess(self, frame):
        try:
            image_raw = frame
            h, w, c = image_raw.shape
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            # image = image_raw
            # Calculate widht and height and paddings
            r_w = self.input_width / w
            r_h = self.input_height / h
            if r_h > r_w:
                tw = self.input_width
                th = int(r_w * h)
                tx1 = tx2 = 0
                ty1 = int((self.input_height - th) / 2)
                ty2 = self.input_height - th - ty1
            else:
                tw = int(r_h * w)
                th = self.input_height
                tx1 = int((self.input_width - tw) / 2)
                tx2 = self.input_width - tw - tx1
                ty1 = ty2 = 0
            # Resize the image with long side while maintaining ratio
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0
            # HWC to CHW format:
            image = np.transpose(image, [2, 0, 1])
            # CHW to NCHW format
            image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order, also known as "C order":
            image = np.ascontiguousarray(image)
            return image, h, w
        except Exception as e:
            logging.error(
                f"{__module_name__} Yolo PreProcessing Failed: {e}")
            logging.error(f"{__module_name__}:\n {traceback.format_exc()}")
            raise Exception(
                f"{__module_name__} Yolo PreProcessing Failed: {e}")

    def post_process(self, output, origin_h, origin_w):

        detections = []  # Each bbox in detections => [x1, y1, x2, y2, conf, class_id]
        # output = np.transpose(np.squeeze(output[0]))
        # rows = output.shape[0]
        # output ->

        confidences = output[:,4]  # array of objectness confidence (25200)
        selected_indices = confidences > self.conf_thresh

        # Now filter info as needed
        bboxes = output[selected_indices, :5]
        confidences = confidences[selected_indices]
        if self.model_yolov8:
            class_confidences = confidences
        else:
            class_confidences = output[selected_indices, 5:]

        # Get the class_id(index) with the highest confidence in row
        result_classid = [
            np.argmax(class_confidences_row)
            for class_confidences_row in class_confidences
        ]

        # Filter based on filter_class_id
        valid_class_indices = []
        if self.filter_class_id != [] and self.filter_class_id is not None:
            for i in range(len(result_classid)):
                valid_class_indices.append(
                    (class_confidences[i][result_classid[i]] > self.conf_thresh)
                    and (result_classid[i] in self.filter_class_id)
                )
            bboxes = bboxes[valid_class_indices, :]
            confidences = confidences[valid_class_indices]
            result_classid = np.array(result_classid)
            result_classid = result_classid[valid_class_indices]

        # Update format
        bboxes = self.xywh2xyxy(origin_h, origin_w, bboxes)

        # Update detections list
        for index, bb in enumerate(bboxes):
            [label, conf] = [result_classid[index], confidences[index]]
            [xmin, ymin, xmax, ymax] = bb[:4]
            # Ensure small negative values do not come through from detections
            xmin, ymin, xmax, ymax = [
                0 if i < 0 else i for i in [xmin, ymin, xmax, ymax]
            ]
            detections.append(np.array([xmin, ymin, xmax, ymax, conf*100, label]))

        # Perform nms
        detections = self.nms(np.array(detections))
        return detections

    def nms(self, boxes):
        overlapThresh = self.nms_thresh
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        conf = boxes[:, 5]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > overlapThresh)[0]))
            )
            
        return boxes[pick].astype("int").tolist()

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        # X, Y, X, Y, Score, CLASS_ID
        norm_w = self.input_width / origin_w
        norm_h = self.input_height / origin_h

        if norm_h > norm_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_height - norm_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_height - norm_w * origin_h) / 2
            y /= norm_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_width - norm_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_width - norm_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= norm_h

        return y
    
    @staticmethod
    def plot_one_box(x, img, color=None, label=None):
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        # line/font thickness
        color = (0, 0, 255)
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


if __name__ == "__main__":
    import time,imutils

    # YoLov5_TRT instance
    # FILTER ID - LIST IDs ONLY WHAT TO DETECT
    #object generalization
    filter_class_id        = "0" # by default set to person detection
    filter_class_id        = [int(s) for s in filter_class_id.split(',')]
    detector = YoloV5_Triton_Client(
        # model_name="yolov8m_person_v2_fp32",
        # model_name="yolov5s_person_v1_fp32",
        model_name="coco_yolov5m_v1_fp32",
        # model_name="yolov8s_coco_v1_fp32",
        detection_confidence=0.5,
        grpc_host="localhost",
        grpc_port=8011,
        nms_thresh=0.7,
        # input_label = "images",
        # output_label = "output0",
        # filter_class_id=[0]
        )
    
    