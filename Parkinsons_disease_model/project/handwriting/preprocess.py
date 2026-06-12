import cv2
import os
import numpy as np

IMG_SIZE = 224
input_dir = "images"
output_dir = "processed_images"
classes = ["SpiralControl", "SpiralPatients"]
aug_per_image = 2


def rotate_image(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def flip_image(img):
    return cv2.flip(img, 1)  

def zoom_image(img, zoom_factor=1.1):
    h, w = img.shape
    center = (w//2, h//2)
    size = int(w / zoom_factor)
    cropped = img[center[1]-size//2:center[1]+size//2, center[0]-size//2:center[0]+size//2]
    return cv2.resize(cropped, (w, h))


def preprocess_image(img, IMG_SIZE=224):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    blur = cv2.GaussianBlur(img, (5,5), 0)
    
    _, img_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresh = 255 - img_thresh 

    kernel = np.ones((3,3), np.uint8)
    img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    img_final = cv2.resize(img_clean, (IMG_SIZE, IMG_SIZE))
    img_final = img_final.astype(np.float32) / 255.0

    return img_final


def run_batch_preprocess():
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "template"), exist_ok=True)

    for cls in classes:
        input_path = os.path.join(input_dir, cls)
        output_path = os.path.join(output_dir, cls)

        for img_name in os.listdir(input_path):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            img_proc = preprocess_image(img, IMG_SIZE)
            cv2.imwrite(os.path.join(output_path, img_name), (img_proc * 255).astype(np.uint8))

            for i in range(aug_per_image):
                aug_img = (img_proc * 255).astype(np.uint8)

                if np.random.rand() < 0.5:
                    aug_img = flip_image(aug_img)
                angle = np.random.uniform(-15, 15)
                aug_img = rotate_image(aug_img, angle)
                zoom_factor = np.random.uniform(1.05, 1.2)
                aug_img = zoom_image(aug_img, zoom_factor)

                aug_name = img_name.split('.')[0] + f"_aug{i}.jpg"
                cv2.imwrite(os.path.join(output_path, aug_name), aug_img)

    template_path = os.path.join(input_dir, "template.jpg")
    template_img = cv2.imread(template_path)

    if template_img is not None:
        template_img_proc = preprocess_image(template_img, IMG_SIZE)
        cv2.imwrite(
            os.path.join(output_dir, "template", "template.jpg"),
            (template_img_proc * 255).astype(np.uint8),
        )
    else:
        print("Warning: Template image not found!")

    print("Preprocessing and augmentation completed successfully.")


if __name__ == "__main__":
    run_batch_preprocess()
