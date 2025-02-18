import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import json
from typing import List, Tuple


PALETTE_CS = [
    [128, 64, 128],  [244, 35, 232],  [70, 70, 70],   [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35],  [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0],     [0, 0, 142],     [0, 0, 70],     [0, 60, 100],
    [0, 80, 100],    [0, 0, 230],     [119, 11, 32],
]
"""
Palette of 19 Cityscapes classes (RGB), index 19 would be [255,255,255] for 'ignore'.
"""

CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
)
NUM_CLASSES = len(CLASSES)  # 19


def apply_palette(
    label_image: np.ndarray,
    palette: List[List[int]] = PALETTE_CS
) -> np.ndarray:
    """
    Apply a color palette to a 2D label image.

    Args:
        label_image: A 2D array of shape (H, W) with integer class IDs.
        palette: A list of colors, each color is [R, G, B].

    Returns:
        A color image (H, W, 3) in uint8 format, where each class
        ID is replaced by its corresponding color in the palette.
    """
    if label_image.ndim != 2:
        raise ValueError("label_image must be a 2D array.")

    # Create a blank color image filled with 255
    colored = np.full((label_image.shape[0], label_image.shape[1], 3), 255, dtype=np.uint8)
    unique_labels = np.unique(label_image)

    for lab in unique_labels:
        if 0 <= lab < NUM_CLASSES:
            colored[label_image == lab, :] = palette[lab]

    return colored


def read_image(
    filename: str,
    dataset_name: str,
    shape_gt: Tuple[int, int]
) -> np.ndarray:
    """
    Read a color image from disk, optionally resizing for certain datasets.

    Args:
        filename: Full path to the image file on disk.
        dataset_name: Name of the dataset (e.g., 'mapillary_vistas_aspect_1.33_train').
        shape_gt: Desired (height, width) for resizing if needed.

    Returns:
        A NumPy array with shape (H, W, 3) in BGR (as per OpenCV).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Image file not found: {filename}")

    image = cv2.imread(filename)  # shape: (H, W, 3) in BGR
    if dataset_name == 'mapillary_vistas_aspect_1.33_train':
        image = cv2.resize(image, (shape_gt[1], shape_gt[0]))  # (width, height)
    return image


def read_gt(
    filename: str,
    dataset_name: str,
    shape_gt: Tuple[int, int]
) -> np.ndarray:
    """
    Read a ground-truth image (grayscale) from disk, optionally resizing.

    Args:
        filename: Path to the ground-truth image.
        dataset_name: Name of the dataset (may influence resizing).
        shape_gt: (height, width) to resize for certain datasets.

    Returns:
        A 2D NumPy array (H, W) of integer labels.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Ground-truth file not found: {filename}")

    gt = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if dataset_name == 'acdc_train_annotations':
        gt[gt==255] = 19
    if dataset_name == 'mapillary_vistas_aspect_1.33_train':
        gt = cv2.resize(gt, (shape_gt[1], shape_gt[0]), interpolation=cv2.INTER_NEAREST)

    return gt


def read_masks(mask_file: str) -> np.ndarray:
    """
    Load SAM-generated binary masks from an .npz file.

    Args:
        mask_file: Path to the .npz containing 'masks'.

    Returns:
        A NumPy array of region masks (dtype=object or structured).
    """
    if not os.path.isfile(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    data = np.load(mask_file, allow_pickle=True)
    if 'masks' not in data:
        raise ValueError(f"'masks' not found in {mask_file}")
    return data['masks']


def save_annotation(
    annotated: np.ndarray,
    fn_annot: str,
    path_output: str,
    dataset_name: str
) -> None:
    """
    Save the annotated image to disk, creating subdirectories if needed.

    Args:
        annotated: A 2D or 3D NumPy array to be saved.
        fn_annot: Full path to the output annotation file (including extension).
        path_output: Base directory for output.
        dataset_name: Name of the dataset (affects subdirectory logic).
    """
    # Example logic for cityscapes
    if dataset_name == 'cityscapes_train':
        city = os.path.basename(fn_annot).split('_')[0]
        city_dir = os.path.join(path_output, city)
        os.makedirs(city_dir, exist_ok=True)

    # Actually write to fn_annot
    cv2.imwrite(fn_annot, annotated)

    # Similar logic for acdc_train_annotations
    if dataset_name == 'acdc_train_annotations':
        parts = fn_annot.split('/')
        if len(parts) > 6:
            weather = parts[4]
            sample_type = parts[5]
            gopro = parts[6]
            city_dir = os.path.join(path_output, weather, sample_type, gopro)
            os.makedirs(city_dir, exist_ok=True)

    # Double-check it was saved
    cv2.imwrite(fn_annot, annotated)


def sort_masks_in_image(
    candidate_masks: np.ndarray,
    sort_by: str,
    rng_seed: int,
    gpu: int
) -> List[dict]:
    """
    Sort the SAM candidate masks according to the chosen method.

    Args:
        candidate_masks: A list/array of dicts, each with keys: 'area', 'segmentation', etc.
        sort_by: One of ['random', 'area', 'Entropy', 'BvsSB'].
        rng_seed: Random seed for 'random' sorting.
        gpu: GPU ID, for 'BvsSB' loading the JSON file with BvsSB scores.

    Returns:
        A list of sorted masks (by chosen criterion).
    """


    if sort_by == 'random':
        rng = np.random.default_rng(rng_seed)
        indices = rng.permutation(len(candidate_masks))
        sorted_masks = [candidate_masks[i] for i in indices]

    elif sort_by == 'area':
        # Sort descending by area
        sorted_masks = sorted(candidate_masks, key=lambda x: x['area'], reverse=True)

    elif sort_by == 'Entropy':
        # 'mean_prob' must be in each mask; we compute the average distribution,
        # then sort by the Shannon entropy of that distribution, descending
        tensor_file = f"tensors_{gpu}.json"
        if not os.path.isfile(tensor_file):
            raise FileNotFoundError(f"Entropy tensor file not found: {tensor_file}")
        
        with open(tensor_file, 'r') as f:
            tensor_scores = json.load(f)
        
        # Pair each mask with a score, then sort descending
        paired = zip(tensor_scores, candidate_masks)
        sorted_pairs = sorted(paired, key=lambda x: x[0], reverse=True)
        sorted_masks = [pair[1] for pair in sorted_pairs]

    elif sort_by == 'BvsSB':
        tensor_file = f"tensors_{gpu}.json"
        if not os.path.isfile(tensor_file):
            raise FileNotFoundError(f"BvsSB tensor file not found: {tensor_file}")

        with open(tensor_file, 'r') as f:
            bvs_sb_scores = json.load(f)

        # Pair each mask with a score, then sort descending
        paired = zip(bvs_sb_scores, candidate_masks)
        sorted_pairs = sorted(paired, key=lambda x: x[0], reverse=True)
        sorted_masks = [pair[1] for pair in sorted_pairs]

        # Remove the JSON once it's used
        os.remove(tensor_file)

    else:
        raise ValueError(f"Unknown sort criterion: {sort_by}")

    return sorted_masks


def make_annotated_image(
    annotated_masks: List[dict],
    shape_gt: Tuple[int, int],
    ignore_label_annotations: int
) -> np.ndarray:
    """
    Create a 2D annotated image from the chosen masks, with region-level labels.

    Args:
        annotated_masks: List of dicts with keys 'area', 'segmentation', 'majority_label_gt'.
        shape_gt: (H, W) shape for the output annotation.
        ignore_label_annotations: Label ID for 'ignore' or background.

    Returns:
        A 2D NumPy array of shape_gt with class IDs in [0..NUM_CLASSES-1] or ignore_label_annotations.
    """
    if len(shape_gt) != 2:
        raise ValueError("shape_gt must be (height, width).")

    annotated_image = np.full(shape_gt, ignore_label_annotations, dtype=np.uint8)
    # Sort by descending area so that smaller overlapping regions appear on top
    sorted_by_area = sorted(annotated_masks, key=lambda x: x['area'], reverse=True)

    for m in sorted_by_area:
        seg = m['segmentation']
        label = m['majority_label_gt']
        if not (0 <= label < NUM_CLASSES):
            raise ValueError(f"Invalid label {label} for region in annotated_masks.")
        annotated_image[seg] = label

    return annotated_image


def make_filenames(
    dataset: str,
    base_filenames: str,
    path_images: str,
    path_groundtruth: str,
    path_masks: str,
    path_output: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Determine file paths for images, ground-truth, masks, and annotation outputs.

    Args:
        dataset: Name of the dataset (cityscapes_train, mapillary_vistas_aspect_1.33_train, acdc_train_annotations).
        base_filenames: The base name(s) (without extension). We assume only one image at a time.
        path_images: Directory with original images.
        path_groundtruth: Directory with labeled groundtruth images.
        path_masks: Directory with .npz files for SAM regions.
        path_output: Base directory for output annotation.

    Returns:
        (fnames_images, fnames_groundtruth, fnames_masks, fnames_annotated):
            Each is a list of file paths (though we only handle one image, they are lists of length 1).
    """
    fn = base_filenames  # We only handle 1 image at a time

    # Build paths depending on dataset
    if dataset == 'cityscapes_train':
        fn_ima = os.path.join(path_images, fn + '_leftImg8bit.png')
        fn_gt = os.path.join(path_groundtruth, fn + '_gtFine_labelTrainIds.png')
        fn_masks_ = os.path.join(path_masks, fn + '_masks.npz')
        fn_annot = os.path.join(path_output, fn + '.png')

    elif dataset == 'mapillary_vistas_aspect_1.33_train':
        fn_ima = os.path.join(path_images, fn + '.jpg')
        fn_gt = os.path.join(path_groundtruth, fn + '.png')
        fn_masks_ = os.path.join(path_masks, fn + '.npz')
        fn_annot = os.path.join(path_output, fn + '.png')

    elif dataset == 'acdc_train_annotations':
        fn_ima = os.path.join(path_images, fn + '_rgb_anon.png')
        fn_gt = os.path.join(path_groundtruth, fn + '_gt_labelTrainIds.png')
        fn_masks_ = os.path.join(path_masks, fn + '_rgb_anon.npz')
        fn_annot = os.path.join(path_output, fn + '_region_annotation.png')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Confirm existence of input files
    for fcheck in [fn_ima, fn_gt, fn_masks_]:
        if not os.path.isfile(fcheck):
            raise FileNotFoundError(f"{fcheck} does not exist.")

    # Confirm we are not overwriting an existing annotation
    if os.path.isfile(fn_annot):
        raise FileExistsError(f"{fn_annot} already exists; not overwriting.")

    return [fn_ima], [fn_gt], [fn_masks_], [fn_annot]


def annotate_masks(
    fn_gt: str,
    fn_masks: str,
    num_masks_to_annotate: int,
    min_area: int,
    ignore_label_groundtruth: int,
    sort_by: str,
    rng_seed: int,
    gpu: int
) -> List[dict]:
    """
    Based on ground-truth and a list of candidate SAM masks, produce an annotation
    for up to `num_masks_to_annotate` largest or highest-criterion regions.

    Args:
        fn_gt: Path to the ground-truth file.
        fn_masks: Path to the .npz with SAM region masks.
        num_masks_to_annotate: Number of masks (regions) we want to annotate.
        min_area: Minimum region area. Smaller regions are skipped.
        ignore_label_groundtruth: ID used for ignoring ground-truth areas.
        sort_by: Sorting criterion (random, area, entropy, BvsSB).
        rng_seed: Random seed for 'random' sorting.
        gpu: GPU ID used in the name of the BvsSB JSON file.

    Returns:
        A list of region dicts, each having 'segmentation' and 'majority_label_gt'.
    """
    gt = read_gt(fn_gt, args.dataset, args.shape_gt)
    masks = read_masks(fn_masks)

    if len(masks) == 0:
        print(f"No masks found in {fn_masks}.")
        return []

    mask_shape = masks[0]['segmentation'].shape
    if gt.shape != mask_shape:
        raise ValueError(
            f"Mismatch in shapes: gt {gt.shape} != segmentation {mask_shape}"
        )

    # Sort masks by the chosen criterion
    sorted_masks = sort_masks_in_image(masks, sort_by, rng_seed, gpu)

    annotated_list = []
    count_annotated = 0
    for m in sorted_masks:
        area = m['area']
        if area <= min_area:
            continue

        seg = m['segmentation']
        counts_gt = np.bincount(gt[seg], minlength=NUM_CLASSES)
        # Argmax might be ignore_label_groundtruth or a normal class
        majority_label_gt = np.argmax(counts_gt)

        # Discard if it's "mostly ignore" region
        if majority_label_gt != ignore_label_groundtruth:
            m['majority_label_gt'] = majority_label_gt
            annotated_list.append(m)
            count_annotated += 1

        if count_annotated == num_masks_to_annotate:
            break

    if count_annotated < num_masks_to_annotate:
        print(f"Only {count_annotated} annotated masks in {fn_masks}, requested {num_masks_to_annotate}.")

    return annotated_list


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for annotation generation.
    """
    parser = argparse.ArgumentParser(
        description="Saves images with some SAM regions annotated "
                    "with the majority of ground-truth in them."
    )
    parser.add_argument('--dataset', required=True,
                        choices=['cityscapes_train', 'mapillary_vistas_aspect_1.33_train', 'acdc_train_annotations'],
                        help='Dataset name. Defines how filenames are constructed.')
    parser.add_argument('--filenames', required=True,
                        help='Base part of the filename for images, ground-truth, masks, etc.')
    parser.add_argument('--path-images', required=True,
                        help='Directory with original color images.')
    parser.add_argument('--path-groundtruth', required=True,
                        help='Directory with ground-truth label images.')
    parser.add_argument('--path-masks', required=True,
                        help='Directory with .npz binary maps for SAM regions.')
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Ignore SAM regions smaller than this area (default=1000).')
    parser.add_argument('--budget', type=int, required=True,
                        help='Number of masks to annotate per image.')
    parser.add_argument('--sort-by', required=True,
                        choices=['random', 'area', 'Entropy', 'BvsSB'],
                        help='Sorting criterion for candidate masks.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default=1234).')
    parser.add_argument('--ignore-label-annotations', type=int, required=True,
                        help='Label ID for ignoring in the final annotated image.')
    parser.add_argument('--path-output', required=True,
                        help='Output directory where annotated images will be saved.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to read BvsSB JSON file (default=0).')

    args = parser.parse_args()

    # If needed, make output dir
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
        print(f"Created output directory: {args.path_output}")

    # Determine shape_gt and ignore_label_groundtruth based on dataset
    if args.dataset == 'cityscapes_train':
        args.ignore_label_groundtruth = 255
        args.shape_gt = (1024, 2048)  # May not always be used
    elif args.dataset == 'mapillary_vistas_aspect_1.33_train':
        args.ignore_label_groundtruth = 19
        args.shape_gt = (1216, 1632)
    elif args.dataset == 'acdc_train_annotations':
        args.ignore_label_groundtruth = 19
        args.shape_gt = (1080, 1920)  # Original shapes
    # else is covered by parser 'choices' => no need for an else.

    if args.ignore_label_groundtruth < NUM_CLASSES:
        raise ValueError("ignore_label_groundtruth must be >= 19 for these datasets.")

    return args


def main() -> None:
    """
    Main entry point for annotation generation. Loads arguments, processes
    a single image (or list with one item) from `--filenames`, sorts SAM regions,
    and writes out the final annotated label image.
    """
    global args
    args = parse_args()

    display_images = False  # set True if you want interactive plotting

    # Build the file lists (all single-element lists in practice)
    fnames_images, fnames_gt, fnames_masks, fnames_annot = make_filenames(
        dataset=args.dataset,
        base_filenames=args.filenames,
        path_images=args.path_images,
        path_groundtruth=args.path_groundtruth,
        path_masks=args.path_masks,
        path_output=args.path_output
    )

    # We handle them in a loop, but normally it's one image at a time.
    for fn_ima, fn_gt, fn_masks, fn_annot in zip(
        fnames_images, fnames_gt, fnames_masks, fnames_annot
    ):
        # Annotate the required number of masks
        annotated_masks = annotate_masks(
            fn_gt=fn_gt,
            fn_masks=fn_masks,
            num_masks_to_annotate=args.budget,
            min_area=args.min_area,
            ignore_label_groundtruth=args.ignore_label_groundtruth,
            sort_by=args.sort_by,
            rng_seed=args.seed,
            gpu=args.gpu
        )

        # Create the final 2D annotated label image
        ann_image = make_annotated_image(
            annotated_masks=annotated_masks,
            shape_gt=args.shape_gt,
            ignore_label_annotations=args.ignore_label_annotations
        )

        # Save to disk
        save_annotation(
            ann_image,
            fn_annot=fn_annot,
            path_output=args.path_output,
            dataset_name=args.dataset
        )

        # Optionally display for debugging
        if display_images:
            ima = read_image(fn_ima, args.dataset, args.shape_gt)
            ima_gt = read_gt(fn_gt, args.dataset, args.shape_gt)

            plt.figure()
            plt.subplot(3, 1, 1)
            plt.imshow(ima[..., ::-1])  # BGR->RGB for display
            plt.axis('off')
            plt.title("Original Image")

            plt.subplot(3, 1, 2)
            plt.imshow(apply_palette(ima_gt))
            plt.axis('off')
            plt.title("Ground Truth")

            plt.subplot(3, 1, 3)
            plt.imshow(apply_palette(ann_image))
            plt.axis('off')
            plt.title("Annotated with SAM Regions")

            plt.show(block=True)


if __name__ == '__main__':
    main()
