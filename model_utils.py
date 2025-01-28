import json
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from huggingface_hub import hf_hub_url, cached_download
from scipy.stats import entropy



def load_label_maps() -> tuple[dict[int, str], dict[str, int], int]:
    """
    Load the label maps for the dataset from Hugging Face Hub.
    """
    repo_id = "huggingface/label-files"
    filename = "cityscapes-id2label.json"
    json_path = cached_download(hf_hub_url(repo_id, filename, repo_type="dataset"))

    with open(json_path, "r") as f:
        id2label_raw = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}
    label2id = {v: k for k, v in id2label.items()}

    id2label[19] = "ignore"
    label2id["ignore"] = 19

    num_labels = len(id2label)
    if num_labels != 20:
        raise ValueError("Number of labels must be 20 for Cityscapes dataset.")
    return id2label, label2id, num_labels


def load_model_and_processor(
    id2label: dict[int, str],
    label2id: dict[str, int],
    num_labels: int,
    device: torch.device
) -> tuple[SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerImageProcessor]:
    """
    Load the segmentation model (SegFormer) and image processor.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        "hector-alvarez/segformerb5-cityscapes",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Resize input images depending on the size of the gpu's
    processor = SegformerImageProcessor(do_resize=True, size={"width": 960, "height": 540})
    evaluating_processor = SegformerImageProcessor(do_resize=False)
    return model, processor, evaluating_processor


def loss_function(
    logits: torch.Tensor,
    lambda_entropy: float,
    inputs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross-entropy + entropy loss.
    """
    if logits is None:
        raise ValueError("Logits cannot be None.")
    if "labels" not in inputs:
        raise ValueError("Inputs dict must contain 'labels'.")

    probs = nn.functional.softmax(logits, dim=1)
    entropy_loss = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=19)(logits, inputs["labels"])
    loss = cross_entropy_loss + lambda_entropy * entropy_loss
    return probs, loss


def calculate_bvs_sb_score(
    probs: torch.Tensor,
    region_masks: list,
    x_reshaped: int,
    y_reshaped: int,
    gpu_number: int
) -> int:
    """
    Calculate best-versus-second-best (BvsSB) for each region.
    """

    if probs is None:
        raise ValueError("Probs cannot be None.")

    regions_bvs_sb_score = []
    for mask in region_masks:
        seg_map = mask["segmentation"]
        image_tensor = torch.from_numpy(seg_map).unsqueeze(0).unsqueeze(0).float()

        resized_tensor = nn.functional.interpolate(
            image_tensor,
            size=(x_reshaped, y_reshaped),
            mode="bilinear",
            align_corners=False
        ).to(probs.device)

        region_probs = probs * resized_tensor
        probs_per_class = torch.mean(region_probs, dim=(2, 3))
        top_values = torch.topk(probs_per_class, 2, dim=1).values
        bvs_sb = top_values[:, 0] - top_values[:, 1]
        regions_bvs_sb_score.append(bvs_sb)

    # Convert to list and save as JSON
    scores_list = [tensor.tolist() for tensor in regions_bvs_sb_score]
    filename = f"tensors_{gpu_number}.json"
    with open(filename, "w") as f:
        json.dump(scores_list, f)

    return 0

def calculate_entropy(
    probs: torch.Tensor,
    region_masks: list,
    x_reshaped: int,
    y_reshaped: int,
    gpu_number: int
) -> int:
    """
    Calculate the entropy of the predictions.
    """

    if probs is None:
        raise ValueError("Probs cannot be None.")

    regions_entropy = []
    for mask in region_masks:
        seg_map = mask["segmentation"]
        image_tensor = torch.from_numpy(seg_map).unsqueeze(0).unsqueeze(0).float()

        resized_tensor = nn.functional.interpolate(
            image_tensor,
            size=(x_reshaped, y_reshaped),
            mode="bilinear",
            align_corners=False
        ).to(probs.device)

        region_probs = probs * resized_tensor
        probs_per_class = torch.mean(region_probs, dim=(2, 3))
        entropies = entropy(probs_per_class.cpu().detach().numpy().T)
        regions_entropy.append(entropies)

    # Convert to list and save as JSON
    scores_list = [tensor.tolist() for tensor in regions_entropy]
    filename = f"tensors_{gpu_number}.json"
    with open(filename, "w") as f:
        json.dump(scores_list, f)
    return 0