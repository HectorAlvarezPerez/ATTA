import os
import numpy as np
from PIL import Image
import torch
from model_utils import loss_function, calculate_bvs_sb_score, calculate_entropy
from data_utils import generate_segmented_mask


def evaluate_first_image(
    model,
    processor,
    img: Image.Image,
    region_ann: list,
    device: torch.device,
    gpu: int,
    sort_by: str
) -> None:
    """
    For the first image, compute BvsSB scores to enable sorting by BvsSB.
    """
    with torch.no_grad():
        unlabeled_inputs = processor(images=[img], return_tensors="pt").to(device)
        model.eval()
        outputs = model(**unlabeled_inputs)

        logits_unlabeled = torch.nn.functional.interpolate(
            outputs.logits,
            size=unlabeled_inputs["pixel_values"].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        probs_unlabeled = torch.nn.functional.softmax(logits_unlabeled, dim=1)

        if sort_by == "BvsSB":
            calculate_bvs_sb_score(
                probs_unlabeled,
                region_ann,
                x_reshaped=unlabeled_inputs["pixel_values"].shape[-2],
                y_reshaped=unlabeled_inputs["pixel_values"].shape[-1],
                gpu_number=gpu
            )
        elif sort_by == "Entropy":
            calculate_entropy(
                probs_unlabeled,
                region_ann,
                x_reshaped=unlabeled_inputs["pixel_values"].shape[-2],
                y_reshaped=unlabeled_inputs["pixel_values"].shape[-1],
                gpu_number=gpu
            )


def train_atta(
    train_filenames: list[str],
    lambda_entropy: float,
    evaluate_every: int,
    num_labels: int,
    model,
    processor,
    optimizer,
    metric,
    device: torch.device,
    tta_type: str,
    base_path_images: str,
    base_path_annotations: str,
    images_extension: str,
    annotations_extension: str,
    region_annotation_extension: str,
    region_annotation_path: str,
    args
) -> None:
    """
    Train the model with ATTA (CTTA/FTTA).
    """
    if len(train_filenames) == 0:
        raise ValueError("No training filenames provided.")

    all_preds, all_labels = [], []

    for i, fname in enumerate(train_filenames):
        print(f"Training on image {i+1}/{len(train_filenames)}")

        img_path = os.path.join(base_path_images, f"{fname}{images_extension}")
        ann_path = os.path.join(base_path_annotations, f"{fname}{annotations_extension}")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        img = Image.open(img_path)
        ann = Image.open(ann_path)

        # Decide how to create active_label
        if tta_type in ["CTTA_upperbound", "FTTA_upperbound"]:
            active_label = Image.fromarray(
                np.minimum(np.array(ann), num_labels - 1),
                mode='L'
            )
        elif tta_type in ["CTTA", "FTTA"]:
            # BvsSB requires region NPZ + scoring
            if args.sort_by == "BvsSB" or args.sort_by == "Entropy":
                region_ann_npz = os.path.join(
                    region_annotation_path,
                    f"{fname}{region_annotation_extension}"
                )
                if not os.path.exists(region_ann_npz):
                    raise FileNotFoundError(f"Region annotation file not found: {region_ann_npz}")

                region_ann = np.load(region_ann_npz, allow_pickle=True)["masks"]
                if i == 0:
                    # Evaluate first image to write out BvsSB JSON file
                    evaluate_first_image(model, processor, img, region_ann, device, args.gpu, args.sort_by)

            # Generate partial annotation externally
            generate_segmented_mask(fname, args)

            out_mask_path = os.path.join(args.path_output, f"{fname}_region_annotation.png")
            active_mask = Image.open(out_mask_path).convert("L")
            active_label = Image.fromarray(
                np.minimum(np.array(active_mask), num_labels - 1),
                mode='L'
            )
        else:
            raise ValueError(f"Invalid TTA type: {tta_type}")

        # Training step
        inputs = processor(
            images=[img],
            segmentation_maps=[active_label],
            return_tensors="pt"
        ).to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=inputs["labels"].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        probs, loss = loss_function(logits, lambda_entropy, inputs)

        # If BvsSB, re-compute region scores after update
        if args.sort_by == "BvsSB":
            calculate_bvs_sb_score(
                probs,
                region_ann,
                x_reshaped=inputs["labels"].shape[1],
                y_reshaped=inputs["labels"].shape[2],
                gpu_number=args.gpu
            )
        if args.sort_by == "Entropy":
            calculate_entropy(
                probs,
                region_ann,
                x_reshaped=inputs["labels"].shape[1],
                y_reshaped=inputs["labels"].shape[2],
                gpu_number=args.gpu
            )

        loss.backward()
        optimizer.step()

        # Track predictions for partial training eval
        logits_np = logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = inputs["labels"].detach().cpu().numpy()
        all_preds.append(logits_np)
        all_labels.append(labels_np)

        # Evaluate every N steps
        if (i + 1) % evaluate_every == 0:
            print("\nEvaluating partial training predictions...")
            model.eval()

            all_preds_concat = np.concatenate(all_preds, axis=0)
            all_labels_concat = np.concatenate(all_labels, axis=0)

            train_metrics = metric.compute(
                predictions=all_preds_concat,
                references=all_labels_concat,
                num_labels=num_labels,
                ignore_index=19
            )
            all_preds.clear()
            all_labels.clear()

            print("Train metrics:", train_metrics, "\n")

        print(f"Loss: {loss.item()}")
