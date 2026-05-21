import os
import shutil

# Paths (edit as needed)
SEG_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_LIST = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'train.txt')
VAL_LIST = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'val.txt')

IMG_SRC = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'images')
BBOX_SRC = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'bbox')
ANNOTATED_SRC = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'annotated_images')
MASK_SRC = os.path.join(SEG_ROOT, 'Kvasir-SEG', 'masks')

TRAIN_IMG_DST = os.path.join(SEG_ROOT, 'train_data', 'images')
TRAIN_BBOX_DST = os.path.join(SEG_ROOT, 'train_data', 'bbox')
TRAIN_ANNOTATED_DST = os.path.join(SEG_ROOT, 'train_data', 'annotated_images')
TRAIN_MASK_DST = os.path.join(SEG_ROOT, 'train_data', 'masks')
VAL_IMG_DST = os.path.join(SEG_ROOT, 'val_data', 'images')
VAL_BBOX_DST = os.path.join(SEG_ROOT, 'val_data', 'bbox')
VAL_ANNOTATED_DST = os.path.join(SEG_ROOT, 'val_data', 'annotated_images')
VAL_MASK_DST = os.path.join(SEG_ROOT, 'val_data', 'masks')

for d in [TRAIN_IMG_DST, TRAIN_BBOX_DST, TRAIN_ANNOTATED_DST, TRAIN_MASK_DST, VAL_IMG_DST, VAL_BBOX_DST, VAL_ANNOTATED_DST, VAL_MASK_DST]:
    os.makedirs(d, exist_ok=True)

def copy_files(list_path, img_dst, bbox_dst, annotated_dst, mask_dst):
    with open(list_path, 'r') as f:
        for line in f:
            base = os.path.splitext(os.path.basename(line.strip()))[0]
            img_file = os.path.join(IMG_SRC, base + '.jpg')
            bbox_file = os.path.join(BBOX_SRC, base + '.csv')
            annotated_file = os.path.join(ANNOTATED_SRC, base + '.jpg')
            mask_file = os.path.join(MASK_SRC, base + '.jpg')
            if os.path.exists(img_file):
                shutil.copy(img_file, img_dst)
            if os.path.exists(bbox_file):
                shutil.copy(bbox_file, bbox_dst)
            if os.path.exists(annotated_file):
                shutil.copy(annotated_file, annotated_dst)
            if os.path.exists(mask_file):
                shutil.copy(mask_file, mask_dst)

if __name__ == '__main__':
    print('Copying training files...')
    copy_files(TRAIN_LIST, TRAIN_IMG_DST, TRAIN_BBOX_DST, TRAIN_ANNOTATED_DST, TRAIN_MASK_DST)
    print('Copying validation files...')
    copy_files(VAL_LIST, VAL_IMG_DST, VAL_BBOX_DST, VAL_ANNOTATED_DST, VAL_MASK_DST)
    print('Done!')
