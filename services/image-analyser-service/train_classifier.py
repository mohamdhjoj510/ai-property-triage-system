"""Train a ResNet18-based real-estate room classifier via transfer learning.

Dataset layout (expected, relative to repo root):
    data/image-dataset/training_dataset_v4_train_test/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg

Classes are inferred automatically from the folder names under train/.
The script:
  1. Loads train/ and test/ via torchvision.datasets.ImageFolder.
  2. Builds a ResNet18 pretrained on ImageNet, freezes the backbone, and
     replaces the final FC layer with one sized for our class count.
  3. Trains only the new classifier head (fast and works well on small data).
  4. Reports train/test accuracy and loss each epoch.
  5. Saves the model checkpoint and the class-to-index mapping.

Run from any CWD — all paths resolve from this file's location:
    python services/image-analyser-service/train_classifier.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --- Paths (resolved from script location, not CWD) ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # services/image-analyser-service/ -> repo root
DATASET_ROOT = REPO_ROOT / "data" / "image-dataset" / "training_dataset_v4_train_test"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

MODELS_DIR = SCRIPT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "real_estate_room_classifier.pth"
CLASS_MAP_PATH = MODELS_DIR / "class_to_idx.json"

# --- Hyperparameters ---
NUM_EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
# Keep at 0 on Windows to avoid multiprocessing pickling issues. Bump on Linux.
NUM_WORKERS = 0
IMG_SIZE = 224  # ResNet's native input
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Train transforms include light augmentation; test transforms do not."""
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfm, test_tfm


def build_dataloaders() -> tuple[datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader]:
    """Load ImageFolder datasets for train/ and test/ and wrap in DataLoaders."""
    train_tfm, test_tfm = build_transforms()
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tfm)
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=test_tfm)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return train_ds, test_ds, train_loader, test_loader


def build_model(num_classes: int) -> nn.Module:
    """ResNet18 pretrained on ImageNet with the final FC replaced.

    The backbone is frozen so only the new head is trained. This is fast,
    requires little data, and avoids over-fitting the backbone on a small
    domain-specific dataset.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze the backbone — gradient updates will only flow through the new head.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the 1000-class ImageNet head with one sized for our classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main() -> None:
    print(f"Dataset root: {DATASET_ROOT}")
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Expected train/ and test/ subdirectories under {DATASET_ROOT}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, test_ds, train_loader, test_loader = build_dataloaders()
    class_names = train_ds.classes
    class_to_idx = train_ds.class_to_idx
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train images: {len(train_ds)}   Test images: {len(test_ds)}")

    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            "train/ and test/ have different class folders. "
            f"train={train_ds.class_to_idx}  test={test_ds.class_to_idx}"
        )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only the FC head has requires_grad=True; pass only those params to the optimiser.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    final_train_loss = final_train_acc = 0.0
    final_test_loss = final_test_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}  train acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f}  test acc {test_acc * 100:.2f}%"
        )
        final_train_loss, final_train_acc = train_loss, train_acc
        final_test_loss, final_test_acc = test_loss, test_acc

    # Save a complete checkpoint — state_dict plus everything inference will need
    # to reconstruct the model (architecture, class mapping, preprocessing stats).
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "num_classes": len(class_names),
            "architecture": "resnet18",
            "img_size": IMG_SIZE,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")

    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)
    print(f"Saved class mapping: {CLASS_MAP_PATH}")

    # Final summary
    print()
    print("=" * 60)
    print("Training complete.")
    print(f"Classes:          {class_names}")
    print(f"Train accuracy:   {final_train_acc * 100:.2f}%")
    print(f"Test accuracy:    {final_test_acc * 100:.2f}%")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final test loss:  {final_test_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
