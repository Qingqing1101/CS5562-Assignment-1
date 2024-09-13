import argparse
import os
from resnet_attack_todo import ResnetPGDAttacker
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np


from huggingface_hub import login

login()

parser = argparse.ArgumentParser(description="Attacking a Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)
parser.add_argument('--alpha', type=float, help='step size for PGD attack', default=2 / 255)
parser.add_argument('--steps', type=int, help='number of steps for PGD attack', default=20)
parser.add_argument('--batch_size', type=int, help='batch size for PGD attack', default=100)
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=None)
parser.add_argument('--results', type=str, help='name of the file to save the results to', required=True)
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='results')
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument('--test', action='store_true', help='test that code runs')
args = parser.parse_args()

RESULTS_DIR = args.resultsdir
RESULTS_PATH = os.path.join(RESULTS_DIR, args.results)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)
else:
    SEED = torch.seed()

EPS = args.eps
ALPHA = args.alpha
STEPS = args.steps

BATCH_SIZE = args.batch_size
BATCH_NUM = args.batch_num
if BATCH_NUM is None:
    BATCH_NUM = 1281167 // BATCH_SIZE + 1
assert BATCH_NUM > 0

print('Loading model...')
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()

# Step 2: Load and preprocess data
print('Loading data...')

# Load ImageNet-1k dataset from Huggingface
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)


def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


# Filter out grayscale images
ds = ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
# Only take desired portion of dataset
ds = ds.take(BATCH_NUM * BATCH_SIZE)

dset_loader = DataLoader(ds, batch_size=BATCH_SIZE)
dset_classes = weights.meta["categories"]
attacker = ResnetPGDAttacker(model, dset_loader)

if args.test:
    print(f"===Testing on {BATCH_NUM if BATCH_NUM else 'all'} batches of data===")
    attacker.compute_accuracy(BATCH_NUM)
    print(f"Accuracy on original images: {attacker.acc * 100}%")
    torch.save({
        'acc': attacker.acc,
    }, RESULTS_PATH)

else:
    print(f"===Launching PGD attack on {BATCH_NUM if BATCH_NUM else 'all'} batches of data===")
    print(f"Attack configs: eps = {EPS}, alpha = {ALPHA}, steps = {STEPS}, batch size = {BATCH_SIZE}")

    # attacker.pgd_batch_attack(EPS, ALPHA, STEPS, BATCH_NUM)
    # print(f"Accuracy on original images: {attacker.acc * 100}%")
    # print(f"Accuracy on adversarial images: {attacker.adv_acc * 100}%")

    # torch.save({
    #     'acc': attacker.acc,
    #     'adv_acc': attacker.adv_acc,
    #     'adv_images': attacker.adv_images,
    #     'labels': attacker.labels,
    # }, RESULTS_PATH)
    
    # Define different epsilon values
    epsilon_values = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    accuracies = []

    # Run PGD attack for each epsilon value
    for eps in epsilon_values:
        print(f"Running PGD attack with epsilon = {eps}")
        attacker.pgd_batch_attack(eps, ALPHA, STEPS, BATCH_NUM)
        print(f"Accuracy on original images: {attacker.acc * 100}%")
        print(f"Accuracy on adversarial images: {attacker.adv_acc * 100}%")
        accuracies.append(attacker.adv_acc * 100)

        # Save results
        torch.save({
            'acc': attacker.acc,
            'adv_acc': attacker.adv_acc,
            'adv_images': attacker.adv_images,
            'labels': attacker.labels,
            'categories': [dset_classes[label] for label in attacker.labels.cpu().numpy()],  # 类别名称
        }, os.path.join(RESULTS_DIR, f'{args.results}_eps_{eps}.pt'))

    # plot epsilon vs accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, accuracies, marker='o', linestyle='-')
    plt.title(f'Model accuracy on adversarial images for various epsilon values\nBatch number {BATCH_NUM}, Batch size {BATCH_SIZE}, Alpha {ALPHA}')
    plt.xlabel('Epsilon values')
    plt.ylabel('Attack accuracy (%)')
    plt.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR, 'epsilon_vs_accuracy.png'))
    plt.show()


