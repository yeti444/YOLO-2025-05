import os
import random
import string

from PIL import Image, ImageDraw, ImageFont

FACES_DIR = "../datasets/faces"
OUTPUT_DIR = "card_gen"


def create_one_card(name):
    # ----------------------------- #
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    template = Image.open("./cardData/card_template.png")

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

    # ----------------------------- #
    filename = str(name) + ".png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    img.save(save_path)


if __name__ == "__main__":
    for i in range(0, 20):
        name = i + 1
        create_one_card(name)
        print(str(name))
