from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests
#============2021-06-18,16点39  先写到这里, 先去搞语音.回来再仔细debug一遍.
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50') # 这个加载的是预处理的配置文件
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
# coco:box: "bbox": [x,y,width,height],左上角和长款.
inputs = feature_extractor(images=[image], annotations=[{"annotations": [{"bbox": [383.99, 391.34, 92.74, 64.52]  , "area": 3488.0849000000007, "iscrowd": 0,  "category_id": 81},{"bbox": [2.99, 3.34, 4.74, 5.52]  , "area": 3488.0849000000007, "iscrowd": 0,  "category_id": 81}]  ,"image_id": 0}], return_tensors="pt")  # annotation里面是多个物体.  , 输入box 是左上点和宽高   跟coco的格式是一样的. 转化完后inputs里面的box是  (center_x, center_y, width, height)

# image_id对应images里面图片的索引

outputs = model(**inputs)
#===========然后玩loss就行了.
# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes
print(logits,bboxes)
usedex=logits.max(2)[1]!=91
usedex2=logits.max(2)[0]>0.7
usedex3=logits[(logits.max(2)[1]!=91) & (logits.max(2)[0]>0.7)]
box=bboxes[(logits.max(2)[1]!=91) & (logits.max(2)[0]>0.7)]

#   logits.max(2)  看分类.

# 分类表:
"""
 "id2label": {
    "0": "N/A",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "N/A",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "N/A",
    "27": "backpack",
    "28": "umbrella",
    "29": "N/A",
    "30": "N/A",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "N/A",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "N/A",
    "67": "dining table",
    "68": "N/A",
    "69": "N/A",
    "70": "toilet",
    "71": "N/A",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "N/A",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush"
  },




"""