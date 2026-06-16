#!/usr/bin/env python3
"""Create per-method sequence score CSVs from Google Drive metrics outputs."""

import argparse
import csv
import os
import time
import urllib.error
import urllib.request
from io import StringIO


METHOD_METRICS_FILE_IDS = {
    "MedSAM3_TEXT_POLYP": "1swO0VTeJOym5maUwkH_3a5QEp4_rKkX9",
    "MedSAM2_YOLO_BOX_BOX_STRIDE10": "1946uNbEpq6vH0xWJBSViAh1gzhVZITyu",
    "MedSAM2_GT_BOX_MASK_STRIDE5": "1QOzl-WULsuuFKUopiECr9avllofv5809",
    "MedSAM2_YOLO_BOX_BOX_FIRST": "1xZ7IZSTodjqcmeO3dtWWN_vhPbq22AFw",
    "MedSAM2_YOLO_BOX_MASK_FIRST": "10iXF1Lai77WaHPSUluKHGCrDugSUZ22y",
    "SAM2_LARGE_GT_BOX_FRAME": "1jNnk0cxGWBZSi-3tSEW5pgQZPzxsynpK",
    "MedSAM2_GT_BOX_MASK_FIRST": "15ai9hj8ozScLafU4BFzvgKu5804L8A0O",
    "MedSAM2_YOLO_BOX_MASK_STRIDE5": "1COxHEhdbTIIbEgDU6UhivnaiTqC2xR3E",
    "MedSAM2_GT_BOX_MASK_STRIDE10": "1nbQzgrF_QKIT00pGb3ssRo_RFyOp-0Ox",
    "MedSAM2_GT_BOX_BOX_STRIDE5": "11TirVl6afn5r_HN1y_IRpPtmHHJDXRa1",
    "MedSAM2_GT_BOX_BOX_FIRST": "1Ot7-fMJogcMQziocv6BhPzSKtUnZljsj",
    "MedSAM2_YOLO_BOX_MASK_STRIDE10": "12Kw_54Sgsr98ypoMN6d0Jx2GrK6vDFTX",
    "YOLO_SAM2_GT_BOX_FRAME": "1Hte8GuwCVauU31SNIpikXvRnK4wPRYz8",
    "MedSAM2_GT_BOX_BOX": "1m0yavxwJp9YYPCiUNlIJ-1FEHFc2BhJM",
    "YOLO_SAM2_YOLO_BOX_FRAME": "1JrERhEP2ydKVKQq2aHUal2B64nTK8fSU",
    "MedSAM2_YOLO_BOX_BOX": "1UrN3BUFha-IrmAG7YQKj09slulddfldn",
    "MedSAM2_YOLO_BOX_BOX_STRIDE5": "1t7MscLYRPcEVx2e8ZL241yp2dOmv89yF",
    "MedSAM2_YOLO_BOX_MASK": "16Sv-Qj750F_rZr3E4DP7krQD7_XgEMmI",
    "MedSAM2_GT_BOX_BOX_STRIDE10": "1IAX7kPaOzK7_hg1fbXyUPyo_pGb69SUU",
    "MedSAM2_GT_BOX_MASK": "11G_uIQ9iROI6tZeWeSNpeRTCtbwHcJi5",
}


def download_drive_csv(file_id: str) -> str:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8-sig")
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {file_id}: {last_error}")


def extract_rows(content: str):
    rows = []
    reader = csv.DictReader(StringIO(content))
    for row in reader:
        if not row.get("seq"):
            continue
        rows.append(
            {
                "sequences": f"seq{str(row['seq']).strip()}",
                "DICE SCORE": row["dice"],
                "IoU score": row["iou"],
            }
        )
    rows.sort(key=lambda row: (float(row["DICE SCORE"]), float(row["IoU score"])), reverse=True)
    return rows


def write_csv(path: str, rows) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequences", "DICE SCORE", "IoU score"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/drive_method_sequence_scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for method, file_id in METHOD_METRICS_FILE_IDS.items():
        content = download_drive_csv(file_id)
        rows = extract_rows(content)
        output_path = os.path.join(args.output_dir, f"{method}.csv")
        write_csv(output_path, rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
