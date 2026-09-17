import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import clip


CODES_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODES_DIR.parents[1]
RESULT_DIR = CODES_DIR / "result"

BASE_SIZE = (440, 140)
BASE_BOXES = [
    (12, 8, 63, 128),
    (69, 8, 124, 128),
    (147, 8, 200, 128),
    (204, 8, 257, 128),
    (261, 8, 314, 128),
    (318, 8, 371, 128),
    (375, 8, 428, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize license plates by matching cropped characters to templates with CLIP.")
    parser.add_argument("--plate_dir", type=str, default=str(ROOT_DIR / "车牌识别" / "License_plate"))
    parser.add_argument("--template_dir", type=str, default=str(ROOT_DIR / "车牌识别" / "Character_templates"))
    parser.add_argument("--weight_path", type=str, default=str(CODES_DIR / "ViT-B-32.pt"))
    parser.add_argument("--output_path", type=str, default=str(RESULT_DIR / "plate_result.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--evaluate_by_filename", action="store_true")
    return parser.parse_args()


def scale_boxes(width, height):
    scale_x = width / BASE_SIZE[0]
    scale_y = height / BASE_SIZE[1]
    return [
        (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )
        for x1, y1, x2, y2 in BASE_BOXES
    ]


def white_mask(image):
    arr = np.array(image.convert("RGB"))
    return ((arr[:, :, 0] > 120) & (arr[:, :, 1] > 120) & (arr[:, :, 2] > 120)).astype(np.uint8)


def tighten_crop(image, pad=4):
    mask = white_mask(image)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image.convert("RGB")

    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, image.size[0])
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad + 1, image.size[1])
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")

    canvas = Image.new("RGB", (crop.size[0] + 8, crop.size[1] + 8), (255, 255, 255))
    canvas.paste(crop, (4, 4))
    return canvas


def encode_images(model, preprocess, device, images):
    with torch.no_grad():
        image_tensor = torch.cat([preprocess(image).unsqueeze(0) for image in images], dim=0).to(device)
        features = model.encode_image(image_tensor)
        return features / features.norm(dim=-1, keepdim=True)


def build_template_bank(model, preprocess, device, template_dir):
    province_names = []
    province_images = []
    alnum_names = []
    alnum_images = []

    for path in sorted(Path(template_dir).glob("*.jpg")):
        template_image = tighten_crop(Image.open(path))
        if len(path.stem) == 1:
            alnum_names.append(path.stem)
            alnum_images.append(template_image)
        else:
            province_names.append(path.stem)
            province_images.append(template_image)

    return {
        "province_names": province_names,
        "province_features": encode_images(model, preprocess, device, province_images),
        "alnum_names": alnum_names,
        "alnum_features": encode_images(model, preprocess, device, alnum_images),
    }


def recognize_plate(model, preprocess, device, banks, plate_path):
    image = Image.open(plate_path).convert("RGB")
    boxes = scale_boxes(*image.size)
    predictions = []

    for index, box in enumerate(boxes):
        crop = tighten_crop(image.crop(box))
        crop_feature = encode_images(model, preprocess, device, [crop])
        if index == 0:
            similarities = crop_feature @ banks["province_features"].T
            predictions.append(banks["province_names"][int(similarities.argmax(dim=-1).item())])
        else:
            similarities = crop_feature @ banks["alnum_features"].T
            predictions.append(banks["alnum_names"][int(similarities.argmax(dim=-1).item())])

    return "".join(predictions)


class PlateRecognizer:
    def __init__(self, template_dir, weight_path, device="cpu"):
        self.template_dir = template_dir
        self.weight_path = weight_path
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(self.weight_path, device=self.device)
        self.model.eval()
        self.template_bank = build_template_bank(self.model, self.preprocess, self.device, self.template_dir)

    def recognize(self, plate_path):
        prediction = recognize_plate(
            model=self.model,
            preprocess=self.preprocess,
            device=self.device,
            banks=self.template_bank,
            plate_path=plate_path,
        )
        return {
            "plate_image": Path(plate_path).name,
            "prediction": prediction,
        }


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, preprocess = clip.load(args.weight_path, device=device)
    model.eval()

    template_bank = build_template_bank(model, preprocess, device, args.template_dir)
    plate_paths = sorted(Path(args.plate_dir).glob("*.jpg"))

    results = []
    right_num = 0
    for plate_path in plate_paths:
        prediction = recognize_plate(model, preprocess, device, template_bank, plate_path)
        item = {
            "plate_image": plate_path.name,
            "prediction": prediction,
        }
        if args.evaluate_by_filename:
            expected = plate_path.stem.split("_")[0]
            item["expected"] = expected
            item["correct"] = prediction == expected
            right_num += int(item["correct"])
        results.append(item)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} plate predictions to {output_path}")
    if args.evaluate_by_filename and results:
        accuracy = right_num / len(results) * 100
        print(f"plate exact-match accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
