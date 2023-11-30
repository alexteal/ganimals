# CHANGES FROM PREVIOUS VERSION:
# FOR INPUT LIST USE [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# ADDED "N" KWARG. IF THIS RUNS TOO SLOW, DECREASE IT. IF IT RUNS TOO FAST, INCREASE IT. HIGHER N MORE ACCURATE RESULTS
import time

# you should "pip install pandas" or whatever pip/conda/idk
# main command is run_pruning_tests(model, test_loader, prune_amounts)
# prune amounts is a list of floats from 0 to 1
# you should just do [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# test loader should be a data loader for test or validation set
# if it takes too long ask me or chatgpt how to shorten the data loader
# it also takes a device parameter but it defaults to cuda so u dont need to specify it

# like slantedtriangularlr, don't forge to add the line:
# from pruning import run_pruning_tests


# import time

# import datasets
import pandas as pd
# from PIL import Image
import torch
# import torchvision.transforms as T
from thop import profile, clever_format
# import torch.nn as nn
from torch.nn.utils import prune


# from torch.utils.data import DataLoader, random_split
# from deit.models import deit_tiny_patch16_224
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from datasets import load_dataset, Dataset
# import onnxruntime as ort


def test(model=None, test_loader=None, device=None):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        # print size of test loader
        print(len(test_loader))
        for images, labels in test_loader:
            images = images.to(device)  # Send images to the appropriate device (CPU/GPU)
            labels = labels.to(device)

            # inference
            predicted = model(images)  # Assuming model returns (x, x_dist)

            # print(predicted.shape)

            total += labels.size(0)
            correct += (predicted.argmax(dim=1) == labels).sum().item()

    accuracy = correct / total
    return accuracy


# def collate_fn(batch):
#     images = []
#     labels = []
#     for item in batch:
#         # The 'image' field is a JpegImageFile object
#         image = item['image']
#         if not isinstance(image, Image.Image):
#             raise TypeError(f"Expected PIL.Image.Image but got {type(image)}")
#
#         # Ensure the image is in RGB format
#         image = image.convert('RGB')
#
#         # Apply the transformations
#         image = transform(image)
#
#         # Check that the image has 3 channels (RGB)
#         if image.shape[0] != 3:
#             raise ValueError(f"Expected image with 3 channels but got {image.shape[0]}")
#
#         labels.append(item['label'])
#         images.append(image)
#
#     # Stack images into a single batch tensor
#     images = torch.stack(images)
#     labels = torch.tensor(labels)
#     return images, labels


def get_macs_and_params(device, model):
    input_tensor = torch.rand(1, 3, 224, 224).to(device)
    macs, params = profile(model, inputs=(input_tensor,))
    return macs, params


def prune_model(model, amount=0.1):
    # Prune the qkv and proj layers in the Attention module
    for block in model.blocks:
        prune.l1_unstructured(block.attn.qkv, name='weight', amount=amount)
        prune.remove(block.attn.qkv, 'weight')
        prune.l1_unstructured(block.attn.proj, name='weight', amount=amount)
        prune.remove(block.attn.proj, 'weight')

    # Prune the fc1 and fc2 layers in the Mlp module
    for block in model.blocks:
        prune.l1_unstructured(block.mlp.fc1, name='weight', amount=amount)
        prune.remove(block.mlp.fc1, 'weight')
        prune.l1_unstructured(block.mlp.fc2, name='weight', amount=amount)
        prune.remove(block.mlp.fc2, 'weight')

    return model


def run_pruning_tests(model, test_loader, prune_amounts, device, n=10):
    if not torch.is_grad_enabled():
        torch.set_grad_enabled(True)

    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    macs, params = get_macs_and_params(device, model)

    macs, params = clever_format([macs, params], "%.3f")

    results = []

    i = 0
    for amount in prune_amounts:
        start_time = time.time()
        pruned_model = prune_model(model, amount=amount)

        accuracy = 0
        for _ in range(n):
            accuracy += test(model=pruned_model, test_loader=test_loader, device=device)
        accuracy /= n

        # Calculate the number of parameters
        nonzero_params = 0
        for param in model.parameters():
            if param.requires_grad:
                nonzero_params += torch.count_nonzero(param).item()

        # ta_ratio = accuracy / inference_time if inference_time > 0 else float('inf')

        # macs, params = get_macs_and_params(device, pruned_model)

        results.append({
            'Prune Amount': amount,
            # 'Inference Time': inference_time,
            'Accuracy': accuracy,
            'Params': nonzero_params,
            # 'Time/Accuracy Ratio': ta_ratio,
            'Accuracy/Param Ratio': accuracy / nonzero_params
        })
        time_taken = time.time() - start_time
        print(
            f"Step {i + 1} / {len(prune_amounts)}, Prune Amount: {amount}, Accuracy: {accuracy}, Time Taken: {time_taken}")
        i += 1

    # Assemble the table
    results_df = pd.DataFrame(results)
    with pd.option_context('display.max_columns', None):
        print(results_df)

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('CUDA is available. Using GPU.')
# else:
#     device = torch.device('cpu')
#     print('CUDA is not available. Using CPU')
#
# transform = T.Compose([
#     T.Resize(256, interpolation=3),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
# ])

# imagenet_val = load_dataset("imagenet-1k", split="validation[:10%]", cache_dir="D:/cache")

# fraction = 0.1
#
# subset_length = int(len(imagenet_val) * fraction)
# remaining_length = len(imagenet_val) - subset_length
#
# # Split the dataset randomly
# subset, _ = random_split(imagenet_val, [subset_length, remaining_length])
#
# test_loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# test_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False, collate_fn=collate_fn)

# model = deit_tiny_patch16_224(pretrained=True)
# model.eval()
# model = model.to(device)
# model.head = nn.Linear(model.embed_dim, 10)

# prune_amounts = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# prune_amounts = [0.0, 0.1, 0.25,  0.4, 0.55]
# prune_amounts = [0.0, 0.1, 0.4]
# run_pruning_tests(model, test_loader, prune_amounts, device)
