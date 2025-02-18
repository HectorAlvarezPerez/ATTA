import os
import subprocess
import argparse
import numpy as np
from PIL import Image
import torch  


def load_dataset(train_images_list_path: str, val_images_list_path: str) -> tuple[list[str], list[str]]:
    """
    Load the training and validation datasets (lists of image identifiers).
    """
    if not os.path.exists(train_images_list_path):
        raise FileNotFoundError(f"Training images list not found: {train_images_list_path}")
    if not os.path.exists(val_images_list_path):
        raise FileNotFoundError(f"Validation images list not found: {val_images_list_path}")

    with open(train_images_list_path, "r") as f:
        train_ids = f.read().splitlines()
    with open(val_images_list_path, "r") as f:
        val_ids = f.read().splitlines()

    if len(train_ids) == 0:
        raise ValueError("No training images found.")
    if len(val_ids) == 0:
        raise ValueError("No validation images found.")

    return train_ids, val_ids


def generate_segmented_mask(filename: str, args: argparse.Namespace) -> None:
    """
    Generate segmented masks using an external script (e.g., sam_region_annotate.py).
    """
    annotate_args = [
        "python",
        f"{args.annotations_script}",
        "--dataset", args.dataset,
        "--filenames", filename,
        "--path-images", args.base_path_images,
        "--path-groundtruth", args.base_path_annotations,
        "--path-masks", args.path_masks,
        "--min-area", str(args.min_area),
        "--budget", str(args.budget),
        "--sort-by", args.sort_by,
        "--seed", str(args.seed),
        "--ignore-label-annotations", str(args.ignore_label_annotations),
        "--path-output", args.path_output,
        "--gpu", str(args.gpu)
    ]

    if not os.path.exists(args.base_path_images):
        raise FileNotFoundError(f"Base path for images not found: {args.base_path_images}")
    if not os.path.exists(args.base_path_annotations):
        raise FileNotFoundError(f"Base path for annotations not found: {args.base_path_annotations}")
    if not os.path.exists(args.path_masks):
        raise FileNotFoundError(f"Path to masks not found: {args.path_masks}")

    subprocess.run(annotate_args, check=True)


def evaluate_model(
    model,
    val_filenames: list[str],
    processor,
    metric,
    device,
    num_labels: int,
    base_path_images: str,
    base_path_annotations: str,
    images_extension: str,
    annotations_extension: str
) -> dict:
    """
    Evaluate the model on the validation set.
    """


    if len(val_filenames) == 0:
        raise ValueError("Validation dataset is empty.")

    if not os.path.exists(base_path_images):
        raise FileNotFoundError(f"Base path for images not found: {base_path_images}")
    if not os.path.exists(base_path_annotations):
        raise FileNotFoundError(f"Base path for annotations not found: {base_path_annotations}")

    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for fname in val_filenames:
            img_path = os.path.join(base_path_images, f"{fname}{images_extension}")
            ann_path = os.path.join(base_path_annotations, f"{fname}{annotations_extension}")

            img = Image.open(img_path)
            ann = Image.open(ann_path)
            ann = np.array(ann)
            ann[ann == 255] = 19  # 255 es el valor de ignore_index en PyTorch


            label = Image.fromarray(np.minimum(ann, num_labels - 1), mode='L')
            
            inputs = processor(images=img, segmentation_maps=label, return_tensors="pt").to(device)
            outputs = model(**inputs)

            logits = outputs.logits

            # Interpolate to match label shape
            logits = torch.nn.functional.interpolate(
                logits,
                size=inputs["labels"].shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            pred = logits.argmax(dim=1).cpu().numpy()
            ref = inputs["labels"].cpu().numpy()

            all_preds.append(pred)
            all_labels.append(ref)

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)


    results = metric.compute(
        predictions=all_preds_np,
        references=all_labels_np,
        num_labels=num_labels,
        ignore_index=19
    )
    return results
