import subprocess
import os
import torch


base_command = "python main.py"

experiments = [
    {
        "gpu": 0,
        "tta_type": "FTTA_upperbound",
        "condition": "night",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_night_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_night_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA_upperbound",
        "model_name": "atta-nvidia-b5-ftta-upperbound-night-original-res-eval",
        "epochs": 1
    },
    {
        "gpu": 1,
        "tta_type": "FTTA_upperbound",
        "condition": "fog",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_fog_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_fog_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA_upperbound",
        "model_name": "atta-nvidia-b5-ftta-upperbound-fog-original-res-eval",
        "epochs": 1
    },
    {
        "gpu": 2,
        "tta_type": "FTTA_upperbound",
        "condition": "rain",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_rain_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_rain_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA_upperbound",
        "model_name": "atta-nvidia-b5-ftta-upperbound-rain-original-res-eval",
        "epochs": 1
    },
    {
        "gpu": 3,
        "tta_type": "FTTA_upperbound",
        "condition": "snow",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_snow_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_snow_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA_upperbound",
        "model_name": "atta-nvidia-b5-ftta-upperbound-snow-original-res-eval",
        "epochs": 1
    },
    # # # gpu 4 is not available
    {
        "gpu": 5,
        "tta_type": "FTTA",
        "condition": "night",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_night_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_night_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA",
        "model_name": "atta-nvidia-b5-ftta-night-random-original-res-eval",
        "path-masks": "./data/masks_0.86_0.92_400/acdc",
        "path-output": "./data/acdc/annotated_train_regions_5", # switch to avoid overwriting
        "annotations_script": "sam_region_annotate.py",
        "dataset": "acdc_train_annotations",
        "min-area": 1000,
        "budget": 16,
        "sort-by": "random",
        "seed": 123,
        "ignore-label-annotations": 19,
        "epochs": 1,  
        "evaluate_every": 200,
        "delete_output_annotations": True,
        "region_annotation_extension": "_rgb_anon.npz",
    },
    {
        "gpu": 6,
        "tta_type": "FTTA",
        "condition": "fog",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_fog_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_fog_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA",
        "model_name": "atta-nvidia-b5-ftta-fog-random-original-res-eval",
        "path-masks": "./data/masks_0.86_0.92_400/acdc",
        "path-output": "./data/acdc/annotated_train_regions_6", # switch to avoid overwriting
        "annotations_script": "sam_region_annotate.py",
        "dataset": "acdc_train_annotations",
        "min-area": 1000,
        "budget": 16,
        "sort-by": "random",
        "seed": 123,
        "ignore-label-annotations": 19,
        "epochs": 1,  
        "evaluate_every": 200,
        "delete_output_annotations": True,
        "region_annotation_extension": "_rgb_anon.npz",
    },
    {
        "gpu": 7,
        "tta_type": "FTTA",
        "condition": "rain",
        "lr": 6e-5,
        "lambda_entropy": 1.0,
        "train_images_list_path": "./data/acdc/train_acdc_rain_images.txt",
        "val_images_list_path": "./data/acdc/val_acdc_rain_images.txt",
        "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
        "base_path_annotations": "./data/acdc/gt_trainval/gt",
        "images_extension": "_rgb_anon.png",
        "annotations_extension": "_gt_labelTrainIds.png",
        "path_to_save_model": "./models/FTTA",
        "model_name": "atta-nvidia-b5-ftta-rain-random-original-res-eval",
        "path-masks": "./data/masks_0.86_0.92_400/acdc",
        "path-output": "./data/acdc/annotated_train_regions_7", # switch to avoid overwriting
        "annotations_script": "sam_region_annotate.py",
        "dataset": "acdc_train_annotations",
        "min-area": 1000,
        "budget": 16,
        "sort-by": "random",
        "seed": 123,
        "ignore-label-annotations": 19,
        "epochs": 1,  
        "evaluate_every": 200,
        "delete_output_annotations": True,
        "region_annotation_extension": "_rgb_anon.npz",
    },
]

def run_experiment(experiment):
    """
    Run an experiment on a specific GPU.

    Args:
        experiment (dict): Dictionary containing experiment configurations.
    """
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = str(experiment["gpu"])
    print(f"Setting CUDA_VISIBLE_DEVICES={env['CUDA_LAUNCH_BLOCKING']}")

    command = [
        "python", "main.py",
        "--gpu", str(experiment["gpu"]),
        "--tta_type", experiment["tta_type"],
        "--lr", str(experiment["lr"]),
        "--lambda_entropy", str(experiment["lambda_entropy"]),
        "--train_images_list_path", experiment["train_images_list_path"],
        "--val_images_list_path", experiment["val_images_list_path"],
        "--base_path_images", experiment["base_path_images"],
        "--base_path_annotations", experiment["base_path_annotations"],
        "--images_extension", experiment["images_extension"],
        "--annotations_extension", experiment["annotations_extension"],
        "--path_to_save_model", experiment["path_to_save_model"],
        "--model_name", experiment["model_name"],
    ]

    if "epochs" in experiment:
        command.extend(["--epochs", str(experiment["epochs"])])
    if "evaluate_every" in experiment:
        command.extend(["--evaluate_every", str(experiment["evaluate_every"])])
    if "delete_output_annotations" in experiment:
        command.extend(["--delete_output_annotations", str(experiment["delete_output_annotations"])])
    if "path-masks" in experiment:
        command.extend(["--path-masks", experiment["path-masks"]])
    if "path-output" in experiment:
        command.extend(["--path-output", experiment["path-output"]])
    if "annotations_script" in experiment:
        command.extend(["--annotations_script", experiment["annotations_script"]])
    if "dataset" in experiment:
        command.extend(["--dataset", experiment["dataset"]])
    if "min-area" in experiment:
        command.extend(["--min-area", str(experiment["min-area"])])
    if "budget" in experiment:
        command.extend(["--budget", str(experiment["budget"])])
    if "sort-by" in experiment:
        command.extend(["--sort-by", experiment["sort-by"]])
    if "seed" in experiment:
        command.extend(["--seed", str(experiment["seed"])])
    if "ignore-label-annotations" in experiment:
        command.extend(["--ignore-label-annotations", str(experiment["ignore-label-annotations"])])
    if "condition" in experiment:
        command.extend(["--condition", experiment["condition"]])

    log_file = os.path.join(experiment["path_to_save_model"], f"{experiment['model_name']}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"Running experiment on GPU {experiment['gpu']} with command: {' '.join(command)}")
    with open(log_file, "w") as log:
        subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    for experiment in experiments:
        run_experiment(experiment)
        print("\n")



    # {
    #     "gpu": 0,
    #     "tta_type": "FTTA",
    #     "condition": "rain",
    #     "lr": 6e-5,
    #     "lambda_entropy": 1.0,
    #     "train_images_list_path": "./data/acdc/train_acdc_rain_images.txt",
    #     "val_images_list_path": "./data/acdc/val_acdc_rain_images.txt",
    #     "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
    #     "base_path_annotations": "./data/acdc/gt_trainval/gt",
    #     "images_extension": "_rgb_anon.png",
    #     "annotations_extension": "_gt_labelTrainIds.png",
    #     "path_to_save_model": "./models/FTTA",
    #     "model_name": "atta-v2-ftta-rain",
    #     "path-masks": "./data/masks_0.86_0.92_400/acdc",
    #     "path-output": "./data/acdc/annotated_train_regions_0", # switch to avoid overwriting
    #     "annotations_script": "annotate.py",
    #     "dataset": "acdc_train_annotations",
    #     "min-area": 1000,
    #     "budget": 16,
    #     "sort-by": "random",
    #     "seed": 123,
    #     "ignore-label-annotations": 255,
    #     "epochs": 10,  
    #     "evaluate_every": 200,
    #     "delete_output_annotations": True,
    #     "region_annotation_extension": "_rgb_anon.npz",
    # },

    #     {
    #     "gpu": 1,
    #     "tta_type": "FTTA_upperbound",
    #     "condition": "fog",
    #     "lr": 6e-5,
    #     "lambda_entropy": 1.0,
    #     "train_images_list_path": "./data/acdc/train_acdc_fog_images.txt",
    #     "val_images_list_path": "./data/acdc/val_acdc_fog_images.txt",
    #     "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
    #     "base_path_annotations": "./data/acdc/gt_trainval/gt",
    #     "images_extension": "_rgb_anon.png",
    #     "annotations_extension": "_gt_labelTrainIds.png",
    #     "path_to_save_model": "./models/FTTA_upperbound",
    #     "model_name": "atta-v2-ftta-upperbound-fog",
    #     "epochs": 10
    # },

    # {
    #     "gpu": 5,
    #     "tta_type": "CTTA",
    #     "condition": "snow",
    #     "lr": 6e-5,
    #     "lambda_entropy": 1.0,
    #     "train_images_list_path": "./data/acdc/train_acdc_images.txt",
    #     "val_images_list_path": "./data/acdc/val_acdc_images.txt",
    #     "base_path_images": "./data/acdc/rgb_anon_trainvaltest/rgb_anon",
    #     "base_path_annotations": "./data/acdc/gt_trainval/gt",
    #     "images_extension": "_rgb_anon.png",
    #     "annotations_extension": "_gt_labelTrainIds.png",
    #     "path_to_save_model": "./models/CTTA",
    #     "model_name": "atta-v2-ctta",
    #     "path-masks": "./data/masks_0.86_0.92_400/acdc",
    #     "path-output": "./data/acdc/annotated_train_regions_5", # switch to avoid overwriting
    #     "annotations_script": "annotate.py",
    #     "dataset": "acdc_train_annotations",
    #     "min-area": 1000,
    #     "budget": 16,
    #     "sort-by": "BvsSB",
    #     "seed": 123,
    #     "ignore-label-annotations": 255,
    #     "epochs": 10,  
    #     "evaluate_every": 200,
    #     "delete_output_annotations": True,
    #     "region_annotation_extension": "_rgb_anon.npz",
    # },



