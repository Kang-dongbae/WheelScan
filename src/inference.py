from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Dict

# 내부 모듈 임포트
from utils import ensure_dir, list_images, device_str, compute_slice_params, measure_text
import config as cfg

# =======================
# [4단계] SAHI 추론 (타일 규칙 동일)
# =======================
def stage4_infer_yolo_with_sahi(
    weights_path: Path,
    cropped_test_split: Path = cfg.CROP_TEST,
    out_dir: Path = cfg.STAGE4_DIR,
    sahi_cfg: Dict = cfg.SAHI_CFG,
    keep_empty: bool = True,
    save_vis: bool = True,
):

    lbl_dir = out_dir / "labels"
    vis_dir = out_dir / "images_vis"
    ensure_dir(lbl_dir)
    if save_vis:
        ensure_dir(vis_dir)

    print("\n=== [4단계] SAHI 추론 (YOLO 포맷 저장) ===")
    print(f" - 가중치: {weights_path}")
    print(f" - 테스트셋: {cropped_test_split}")
    print(f" - 출력: {out_dir}")

    dmodel = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=sahi_cfg.get("conf_thres", 0.5),
        device=device_str(),
    )

    imgs = list_images(cropped_test_split / "images")
    if not imgs:
        raise FileNotFoundError(f"크롭 테스트 이미지가 없습니다: {cropped_test_split/'images'}")

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
    except Exception:
        font = None

    for ip in imgs:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, sahi_cfg)

        res = get_sliced_prediction(
            image=im,
            detection_model=dmodel,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=ovh,
            overlap_width_ratio=ovw,
            postprocess_type=sahi_cfg.get("postprocess", "NMS"),
            postprocess_match_metric=sahi_cfg.get("match_metric", "IOU"),
            postprocess_match_threshold=sahi_cfg.get("match_thres", 0.45),
            postprocess_class_agnostic=True,
        )
       
        stem = Path(ip).stem
        yolo_lines = []

        for op in res.object_prediction_list:
            x1, y1, x2, y2 = map(float, op.bbox.to_voc_bbox())
            bw = max(1e-6, x2 - x1)
            bh = max(1e-6, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cxn = min(max(cx / W, 0.0), 1.0)
            cyn = min(max(cy / H, 0.0), 1.0)
            bwn = min(max(bw / W, 1e-6), 1.0)
            bhn = min(max(bh / H, 1e-6), 1.0)

            cls_id = getattr(op.category, "id", 0)
            try:
                cls_id = int(cls_id)
            except Exception:
                cls_id = 0
            score = getattr(op.score, "value", None)
            conf = 0.0 if score is None else float(score)

            yolo_lines.append(f"{cls_id} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f} {conf:.4f}")

        if yolo_lines:
            yolo_lines = sorted(
                yolo_lines,
                key=lambda s: float(s.strip().split()[-1]),
                reverse=True,
            )

        out_txt = lbl_dir / f"{stem}.txt"
        if yolo_lines or keep_empty:
            with open(out_txt, "w") as f:
                f.write("\n".join(yolo_lines))

        if save_vis:
            try:
                vis = im.copy()
                draw = ImageDraw.Draw(vis)
                for op in res.object_prediction_list:
                    x1, y1, x2, y2 = map(int, op.bbox.to_voc_bbox())
                    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=6)

                    cls_name = getattr(op.category, "name", None)
                    cls_id = getattr(op.category, "id", None)
                    label = str(cls_name if cls_name is not None else cls_id)
                    score = getattr(op.score, "value", None)
                    if score is not None:
                        label = f"{label}:{score:.2f}"

                    if label:
                        tw, th = measure_text(draw, label, font=font)
                        tx, ty = x1, max(0, y1 - th - 2)
                        draw.rectangle([(tx, ty), (tx + tw + 2, ty + th + 2)], fill=(255, 0, 0))
                        draw.text((tx + 1, ty + 1), label, fill=(255, 255, 255), font=font)

                vis.save((vis_dir / f"{stem}.png"))
            except Exception as e:
                print(f"⚠️ {ip.name} 시각화 저장 중 오류: {e}")

    print(f"[4단계 완료] YOLO labels: {lbl_dir}")
    if save_vis:
        print(f"[4K단계 완료] 시각화: {vis_dir}")