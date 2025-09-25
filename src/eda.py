# 파일 경로 설정
data_root = r"/content/drive/MyDrive/코레일 AI전문가과정2/WheelScan/data"
train_img_dir = os.path.join(data_root, "train", "images")
coco_train_json = os.path.join(data_root, "train", "labels", "coco", "train.json")
coco_valid_json = os.path.join(data_root, "train", "labels", "coco", "valid.json")

# COCO API 로드
coco_train = COCO(coco_train_json)