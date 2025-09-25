import os
from pycocotools.coco import COCO

# 파일 경로 설정
data_root = r"C:\Dev\WheelScan\data"
train_img_dir = os.path.join(data_root, "train", "images")
train_label_dir = os.path.join(data_root, "train", "labels")


def create_dataframe(yolo):
    # 이미지 정보
    images = []
    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        images.append({
            'image_id': img_info['id'],
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        })
    img_df = pd.DataFrame(images)

    # 어노테이션 정보
    annotations = []
    for ann_id in coco.anns:
        ann = coco.anns[ann_id]
        bbox = ann['bbox']  # [x, y, width, height]
        annotations.append({
            'ann_id': ann_id,
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'category_name': coco.cats[ann['category_id']]['name'],
            'bbox_x': bbox[0],
            'bbox_y': bbox[1],
            'bbox_width': bbox[2],
            'bbox_height': bbox[3],
            'area': bbox[2] * bbox[3]
        })
    ann_df = pd.DataFrame(annotations)

    # 이미지와 어노테이션 병합
    df = pd.merge(ann_df, img_df, on='image_id', how='left')

    return df

# Train 데이터프레임 생성
train_df = create_dataframe(train_label_dir)
print("Train DataFrame Head:")
print(train_df.head(5))
print("\nTrain DataFrame Info:")
print(train_df.info())
