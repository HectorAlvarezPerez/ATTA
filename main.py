import os
import shutil
import argparse
import torch
import evaluate

# Local module imports
from data_utils import load_dataset, evaluate_model
from model_utils import load_label_maps, load_model_and_processor
from train import train_atta


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ATTA training.
    """
    parser = argparse.ArgumentParser(
        description="Train a model using ATTA (Active Test-Time Adaptation)."
    )

    # Required arguments
    parser.add_argument("--gpu", type=int,
                        help="GPU to use for training.")
    parser.add_argument(
        "--tta_type",
        type=str,
        required=True,
        choices=["FTTA", "FTTA_upperbound", "CTTA", "CTTA_upperbound"],
        help="Type of Test-Time Adaptation to use."
    )
    parser.add_argument("--lr", type=float, required=True, 
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lambda_entropy", type=float, required=True, 
                        help="Weight for the entropy loss term.")
    parser.add_argument("--train_images_list_path", type=str, required=True,
                        help="Path to the file containing the list of training image basenames.")
    parser.add_argument("--val_images_list_path", type=str, required=True,
                        help="Path to the file containing the list of validation image basenames.")
    parser.add_argument("--base_path_images", type=str, required=True,
                        help="Base path for the images.")
    parser.add_argument("--base_path_annotations", type=str, required=True,
                        help="Base path for the annotations.")
    parser.add_argument("--path_to_save_model", type=str, required=True,
                        help="Directory to save the trained model weights.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Filename (without .pth) for saving model weights.")
    parser.add_argument(
        "--region_annotation_extension",
        default="_rgb_anon.npz",
        help="Extension of the region annotation files from SAM (default=_rgb_anon.npz)."
    )

    # Optional arguments
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["fog", "night", "rain", "snow"],
        help="Weather condition for FTTA (optional)."
    )
    parser.add_argument("--images_extension", type=str, default="_rgb_anon.png",
                        help="Extension of the image files (default=_rgb_anon.png).")
    parser.add_argument("--annotations_extension", type=str, default="_gt_labelTrainIds.png",
                        help="Extension of the annotation files (default=_gt_labelTrainIds.png).")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of epochs for training (default=1).")
    parser.add_argument("--evaluate_every", type=int, default=100, 
                        help="Frequency of partial training evaluation (default=100).")

    # Annotate script-specific arguments
    parser.add_argument("--annotations_script", type=str, default="sam_region_annotate.py", 
                        help="Path to the script for generating segmented masks.")
    parser.add_argument("--dataset", type=str, default="acdc_train_annotations",
                        help="Name of the dataset to use for annotation (default=acdc_train_annotations).")
    parser.add_argument("--path-masks", type=str,
                        help="Path to the SAM masks for annotation.")
    parser.add_argument("--min-area", type=int, default=1000,
                        help="Minimum region area for annotation (default=1000).")
    parser.add_argument("--budget", type=int, default=16,
                        help="Number of region annotations per image (default=16).")
    parser.add_argument("--sort-by", type=str,
                        help="Sorting criterion for region selection (e.g., BvsSB, random).")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed (default=123).")
    parser.add_argument("--ignore-label-annotations", type=int, default=255,
                        help="Ignore label for annotations (default=255).")
    parser.add_argument("--path-output", type=str,
                        help="Directory to save annotation outputs.")
    parser.add_argument("--delete_output_annotations", type=bool, default=True,
                        help="Delete annotation outputs after training.")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function to train the segmentation model using ATTA.
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label maps + model + processor
    id2label, label2id, num_labels = load_label_maps()
    model, processor, evaluating_processor = load_model_and_processor(id2label, label2id, num_labels, device)

    # Setup optimizer and evaluation metric
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = evaluate.load("mean_iou")

    # Load train and val image lists
    train_ids, val_ids = load_dataset(args.train_images_list_path, args.val_images_list_path)

    # Train for 'epochs'
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_atta(
            train_filenames=train_ids,
            lambda_entropy=args.lambda_entropy,
            evaluate_every=args.evaluate_every,
            num_labels=num_labels,
            model=model,
            processor=processor,
            optimizer=optimizer,
            metric=metric,
            device=device,
            tta_type=args.tta_type,
            base_path_images=args.base_path_images,
            base_path_annotations=args.base_path_annotations,
            images_extension=args.images_extension,
            annotations_extension=args.annotations_extension,
            region_annotation_extension=args.region_annotation_extension,
            region_annotation_path=args.path_masks,
            args=args
        )

        # Validation
        val_metrics = evaluate_model(
            model=model,
            val_filenames=val_ids,
            processor=processor,
            metric=metric,
            device=device,
            num_labels=num_labels,
            base_path_images=args.base_path_images,
            base_path_annotations=args.base_path_annotations,
            images_extension=args.images_extension,
            annotations_extension=args.annotations_extension
        )
        print(f"Validation after epoch {epoch + 1}: {val_metrics}")

        # Possibly remove annotation outputs
        if args.delete_output_annotations and args.path_output:
            if os.path.exists(args.path_output):
                try:
                    shutil.rmtree(args.path_output)
                    print(f"Deleted output directory: {args.path_output}")
                except Exception as e:
                    print(f"Error deleting {args.path_output}: {e}")
            else:
                print(f"No directory found at: {args.path_output}")

    # Save final model
    os.makedirs(args.path_to_save_model, exist_ok=True)
    save_path = os.path.join(args.path_to_save_model, f"{args.model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved at: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    print("\nRunning with arguments:")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    print()

    main(args)
