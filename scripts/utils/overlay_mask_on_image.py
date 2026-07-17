"""
overlay_mask_on_image.py

Hàm chồng (overlay) mask segmentation lên ảnh gốc để trực quan hóa kết quả dự đoán.
"""

import cv2
import numpy as np
import os


def overlay_mask_on_image(
    image_path: str,
    mask_path: str,
    output_path: str = None,
    color: tuple = (0, 0, 255),   # BGR - đỏ, dùng cho vùng polyp
    alpha: float = 0.01,           # độ trong suốt của mask khi chồng lên ảnh
    draw_contour: bool = True,     # có vẽ viền contour quanh mask hay không
    contour_color: tuple = (0, 255, 0),  # BGR - xanh lá cho viền
    contour_thickness: int = 2,
    mask_threshold: int = 127,     # ngưỡng nhị phân hóa mask (nếu mask không phải 0/255)
):
    """
    Chồng mask (grayscale/binary) lên ảnh gốc (RGB/BGR) để visualize kết quả predicted mask.

    Args:
        image_path: đường dẫn ảnh gốc (.jpg/.png/...)
        mask_path: đường dẫn mask dự đoán (.png, grayscale, giá trị 0/255 hoặc 0/1)
        output_path: nếu có, lưu ảnh kết quả ra file này. Nếu None thì không lưu, chỉ trả về array.
        color: màu BGR dùng để tô vùng mask
        alpha: hệ số blend (0 = không thấy mask, 1 = mask che kín màu gốc)
        draw_contour: có vẽ thêm đường viền quanh vùng mask không
        contour_color: màu BGR của đường viền
        contour_thickness: độ dày đường viền
        mask_threshold: ngưỡng để nhị phân hóa mask trước khi overlay

    Returns:
        overlay_img: ảnh kết quả dạng numpy array (BGR), đã chồng mask lên ảnh gốc.
    """
    # Đọc ảnh gốc
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh gốc: {image_path}")

    # Đọc mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {mask_path}")

    # Resize mask về đúng kích thước ảnh gốc nếu lệch nhau
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Nhị phân hóa mask
    binary_mask = (mask >= mask_threshold).astype(np.uint8)

    # Tạo layer màu cho vùng mask
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:] = color

    # Blend: chỉ blend tại các pixel thuộc mask, giữ nguyên ảnh gốc ở phần còn lại
    overlay_img = image.copy()
    mask_bool = binary_mask.astype(bool)
    overlay_img[mask_bool] = cv2.addWeighted(
        image, 1 - alpha, color_layer, alpha, 0
    )[mask_bool]

    # Vẽ contour quanh vùng mask cho dễ nhìn ranh giới
    if draw_contour:
        contours, _ = cv2.findContours(
            binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_img, contours, -1, contour_color, contour_thickness)

    # Lưu file nếu có yêu cầu
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay_img)

    return overlay_img


def overlay_mask_on_image_batch(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
    **overlay_kwargs,
):
    """
    Chạy overlay_mask_on_image cho toàn bộ 1 sequence (vd: seq1), khớp file theo
    tên số thứ tự (0, 1, 2, ...) giữa thư mục ảnh và thư mục mask.

    Args:
        images_dir: thư mục chứa ảnh gốc, vd .../seq1/images
        masks_dir: thư mục chứa mask dự đoán, vd .../gated/masks/seq1/predicted
        output_dir: thư mục lưu ảnh overlay kết quả
        image_ext: đuôi file ảnh gốc (mặc định .jpg)
        mask_ext: đuôi file mask (mặc định .png)
        **overlay_kwargs: các tham số khác truyền tiếp vào overlay_mask_on_image
            (color, alpha, draw_contour, contour_color, contour_thickness, mask_threshold)

    Returns:
        list các output_path đã lưu thành công.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách mask, sort theo số thứ tự (không phải theo string)
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(mask_ext)]
    mask_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    saved_paths = []
    skipped = []

    for mask_file in mask_files:
        stem = os.path.splitext(mask_file)[0]  # vd "0"
        image_file = f"{stem}{image_ext}"

        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        output_path = os.path.join(output_dir, f"{stem}.png")

        if not os.path.exists(image_path):
            skipped.append(stem)
            continue

        overlay_mask_on_image(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            **overlay_kwargs,
        )
        saved_paths.append(output_path)

    print(f"Đã xử lý {len(saved_paths)}/{len(mask_files)} ảnh. Lưu tại: {output_dir}")
    if skipped:
        print(f"Bỏ qua {len(skipped)} file (không tìm thấy ảnh gốc tương ứng): {skipped}")

    return saved_paths


def overlay_mask_on_image_all_sequences(
    images_root: str,
    masks_root: str,
    output_root: str,
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
    masks_subdir: str = "predicted",
    **overlay_kwargs,
):
    """
    Chạy overlay cho TẤT CẢ các sequence (seq1, seq2, seq3, ...) nằm dưới masks_root.

    Cấu trúc thư mục kỳ vọng:
        images_root/<seq>/images/<i>.jpg
        masks_root/<seq>/<masks_subdir>/<i>.png
        output_root/overlay/<seq>/<i>.png   (sẽ được tạo ra)

    Ví dụ với dữ liệu của bạn:
        images_root = "/home/quangdung/AdeSEG/data/test/polypgen"
        masks_root  = "/home/quangdung/AdeSEG/ReliabilityGated_GT_BOX_BOX_STRIDE10/gated/masks"
        output_root = "/home/quangdung/AdeSEG/data/overlay"
        masks_subdir = "predicted"

    Args:
        images_root: thư mục gốc chứa các seq ảnh, mỗi seq có subfolder "images"
        masks_root: thư mục gốc chứa các seq mask, mỗi seq có subfolder masks_subdir
        output_root: thư mục gốc để lưu kết quả, mỗi seq sẽ có 1 subfolder riêng
        image_ext / mask_ext: đuôi file ảnh / mask
        masks_subdir: tên thư mục con chứa mask dự đoán bên trong mỗi seq (mặc định "predicted")
        **overlay_kwargs: truyền tiếp vào overlay_mask_on_image (color, alpha, draw_contour, ...)

    Returns:
        dict {seq_name: [danh sách output_path]}
    """
    if not os.path.isdir(masks_root):
        raise NotADirectoryError(f"masks_root không tồn tại: {masks_root}")

    seq_names = sorted(
        d for d in os.listdir(masks_root)
        if os.path.isdir(os.path.join(masks_root, d))
    )

    if not seq_names:
        print(f"Không tìm thấy sequence nào trong {masks_root}")
        return {}

    results = {}
    for seq in seq_names:
        seq_masks_dir = os.path.join(masks_root, seq, masks_subdir)
        seq_images_dir = os.path.join(images_root, seq, "images")
        seq_output_dir = os.path.join(output_root, seq)

        if not os.path.isdir(seq_masks_dir):
            print(f"[{seq}] Bỏ qua - không tìm thấy thư mục mask: {seq_masks_dir}")
            continue
        if not os.path.isdir(seq_images_dir):
            print(f"[{seq}] Bỏ qua - không tìm thấy thư mục ảnh: {seq_images_dir}")
            continue

        print(f"[{seq}] Đang xử lý...")
        saved_paths = overlay_mask_on_image_batch(
            images_dir=seq_images_dir,
            masks_dir=seq_masks_dir,
            output_dir=seq_output_dir,
            image_ext=image_ext,
            mask_ext=mask_ext,
            **overlay_kwargs,
        )
        results[seq] = saved_paths

    total = sum(len(v) for v in results.values())
    print(f"\nHoàn tất: {total} ảnh overlay trên {len(results)} sequence. Kết quả tại: {output_root}")
    return results


if __name__ == "__main__":

    IMAGE_PATH = "/home/quangdung/AdeSEG/data/test/polypgen/seq1/images/0.jpg"
    MASK_PATH = "/home/quangdung/AdeSEG/ReliabilityGated_GT_BOX_BOX_STRIDE10/gated/masks/seq1/predicted/0.png"
    OUTPUT_PATH = "/home/quangdung/AdeSEG/data/overlay/seq1/0.png"

    result = overlay_mask_on_image(
        image_path=IMAGE_PATH,
        mask_path=MASK_PATH,
        output_path=OUTPUT_PATH,
    )
    print(f"Đã lưu overlay tại: {OUTPUT_PATH}, shape={result.shape}")

    # --- Xử lý toàn bộ sequence (bỏ comment để chạy) ---
    # overlay_mask_on_image_batch(
    #     images_dir="/home/quangdung/AdeSEG/data/test/polypgen/seq1/images",
    #     masks_dir="/home/quangdung/AdeSEG/ReliabilityGated_GT_BOX_BOX_STRIDE10/gated/masks/seq1/predicted",
    #     output_dir="/home/quangdung/AdeSEG/outputs/overlay/seq1",
    # )

    #--- Xử lý TẤT CẢ sequence (seq1, seq2, ...) cùng lúc (bỏ comment để chạy) ---
    overlay_mask_on_image_all_sequences(
        images_root="/home/quangdung/AdeSEG/data/test/polypgen",
        masks_root="/home/quangdung/AdeSEG/ReliabilityGated_GT_BOX_BOX_STRIDE10/gated/masks",
        output_root="/home/quangdung/AdeSEG/data/overlay",
    )