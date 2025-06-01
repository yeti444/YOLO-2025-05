import os
import random
import string

import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torchvision.transforms import ToPILImage, ToTensor

FACES_DIR = "../datasets/faces"
BACKGROUND_DIR = "../datasets/backgrounds"
TEMPLATE = "./cardData/card_template.png"
OUTPUT = "../datasets/student_card"
CLASS_ID = 0


def create_one_card():
    # ----------------------------- #

    template = Image.open(TEMPLATE)

    # ----------------------------- #
    folder_list = []
    for folder_name in os.listdir(FACES_DIR):
        full_path = os.path.join(FACES_DIR, folder_name)
        if os.path.isdir(full_path):
            folder_list.append(folder_name)

    chosen_folder = random.choice(folder_list)
    random_path = os.path.join(FACES_DIR, chosen_folder)

    image_list = []
    for filename in os.listdir(random_path):
        lower = filename.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            image_list.append(filename)

    chosen_image = random.choice(image_list)
    id_path = os.path.join(random_path, chosen_image)

    student_photo = Image.open(id_path).resize((140, 160))

    # ----------------------------- #
    card_number = "H" + str(random.randint(10000000, 99999999))

    characters = string.ascii_letters + string.digits
    random_chars = random.choices(characters, k=6)
    student_number = ""
    for ch in random_chars:
        student_number = student_number + ch.upper()

    student_name = chosen_folder.replace("_", " ").upper()

    # ----------------------------- #
    img = template.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 18)

    draw.text((165, 230), card_number, font=font, fill=(0, 0, 0))
    draw.text((165, 260), student_number, font=font, fill=(0, 0, 0))
    draw.text((70, 297), student_name, font=font, fill=(0, 0, 0))
    img.paste(student_photo, (390, 120))

    return img


def place_card_on_background(card_img, name):
    # ----------------------------- #
    if name % 3 == 0:
        folder = "valid" if (name % 2 == 0) else "test"
    else:
        folder = "train"
    image_dir = os.path.join(OUTPUT, folder, "images")
    label_dir = os.path.join(OUTPUT, folder, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    # ----------------------------- #
    all_background = [
        f
        for f in os.listdir(BACKGROUND_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    background_name = random.choice(all_background)
    background_path = os.path.join(BACKGROUND_DIR, background_name)
    background = Image.open(background_path).convert("RGBA")
    min_size = 640
    w, h = background.size
    scale = max(min_size / w, min_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    background = background.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    random_crop = torchvision.transforms.RandomCrop(640)
    background = random_crop(background)
    card = card_img.convert("RGBA")
    # ----------------------------- #
    new_width = int(background.width * random.uniform(0.2, 0.7))
    new_height = int(card.height * (new_width / card.width))
    card_resized = card.resize(
        (new_width, new_height), resample=Image.Resampling.BILINEAR
    )
    # ----------------------------- #
    card_rotated = torchvision.transforms.functional.rotate(
        card_resized, random.uniform(-180, 180), expand=True
    )
    output_card = torchvision.transforms.RandomPerspective(
        distortion_scale=0.5, p=0.5
    )(card_rotated)
    # ----------------------------- #
    xmax = background.width - output_card.width
    ymax = background.height - output_card.height
    if xmax <= 0 or ymax <= 0:
        scale = min(
            background.width / output_card.width,
            background.height / output_card.height
        ) * 0.9
        output_card = output_card.resize(
            (
                int(output_card.width * scale),
                int(output_card.height * scale)
            ),
            resample=Image.Resampling.BILINEAR
        )
        xmax = background.width - output_card.width
        ymax = background.height - output_card.height
    # ----------------------------- #
    x = random.randint(0, xmax)
    y = random.randint(0, ymax)
    output = background.copy()
    output.paste(output_card, (x, y), output_card)
    # ----------------------------- # 
    deepfry = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(output)
    output = enhancer.enhance(deepfry)
    # ----------------------------- #
    blur = random.uniform(0.0, 2.0)
    output = output.filter(ImageFilter.GaussianBlur(blur))
    # ----------------------------- #
    jitter = torchvision.transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    )
    output = jitter(output)
    # ----------------------------- #
    tensor_image = ToTensor()(output).unsqueeze(0) 
    noise = torch.randn_like(tensor_image) * random.uniform(0, 0.1)
    tensor_img = tensor_image + noise
    tensor_img = torch.clamp(tensor_img, 0.0, 1.0)
    output = ToPILImage()(tensor_img.squeeze(0))
    # ----------------------------- #
    out_img_name = f"{CLASS_ID}_{name}.png"
    out_img_path = os.path.join(image_dir, out_img_name)
    output.convert("RGB").save(out_img_path, "PNG")
    # ----------------------------- #
    alpha = np.array(output_card.split()[-1])
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return
    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0)
    b_xmin = x + x0
    b_xmax = x + x1
    b_ymin = y + y0
    b_ymax = y + y1
    xb = (b_xmin + b_xmax) / 2 / background.width
    yb = (b_ymin + b_ymax) / 2 / background.height
    wb = (b_xmax - b_xmin) / background.width
    wh = (b_ymax - b_ymin) / background.height
    # ----------------------------- #
    label_name = f"{CLASS_ID}_{name}.txt"
    label_path = os.path.join(label_dir, label_name)
    with open(label_path, "w", newline="\n") as line:
        line.write(f"{CLASS_ID} {xb:.6f} {yb:.6f} {wb:.6f} {wh:.6f}")
if __name__ == "__main__":
    img = 2000
    for i in range(1, img + 1):
        place_card_on_background(create_one_card(), i)
        print(f"{i} / {img}")
