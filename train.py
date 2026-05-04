"""
Pokemon Classification with Transfer Learning  -  Homework #6
==============================================================
Based on lecture: Image Classification: CNN Backbones and Transfer Learning

4 Experiment configurations:
  exp1  ResNet-34  | pretrained=True  | freeze=True   (feature extraction)
  exp2  ResNet-34  | pretrained=True  | freeze=False  (full fine-tuning)
  exp3  ResNet-34  | pretrained=False | freeze=False  (train from scratch)
  exp4  EfficientNet-B0 | pretrained=True | freeze=True (feature extraction)

Usage:
  python train.py --data_dir ./PokemonData --exp exp1
  python train.py --data_dir ./PokemonData --all
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------------------------------- #
# Experiment configurations
# Lecture ref: Transfer Learning guidelines (similar/small -> freeze backbone,
#              similar/large -> full fine-tune, different -> vary layers)
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    'exp1': {
        'description': 'ResNet-34 | pretrained=True | freeze=True  (feature extraction)',
        'backbone':    'resnet34',
        'pretrained':  True,
        'freeze':      True,
        'lr':          1e-3,
        'epochs':      20,
        'batch_size':  32,
    },
    'exp2': {
        'description': 'ResNet-34 | pretrained=True | freeze=False (full fine-tuning)',
        'backbone':    'resnet34',
        'pretrained':  True,
        'freeze':      False,
        'lr':          1e-4,   # smaller LR for fine-tuning (lecture tip)
        'epochs':      20,
        'batch_size':  32,
    },
    'exp3': {
        'description': 'ResNet-34 | pretrained=False | freeze=False (train from scratch)',
        'backbone':    'resnet34',
        'pretrained':  False,
        'freeze':      False,
        'lr':          1e-3,
        'epochs':      30,     # more epochs needed without pretrained weights
        'batch_size':  32,
    },
    'exp4': {
        'description': 'EfficientNet-B0 | pretrained=True | freeze=True (feature extraction)',
        'backbone':    'efficientnet_b0',
        'pretrained':  True,
        'freeze':      True,
        'lr':          1e-3,
        'epochs':      20,
        'batch_size':  32,
    },
}


# --------------------------------------------------------------------------- #
# Dataset
# Lecture ref: AlexNet (random crop, hflip, RGB jitter),
#              VGGNet  (1 image -> 150 augmented images)
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_datasets(data_dir: str, val_ratio: float = 0.15,
                  test_ratio: float = 0.15, seed: int = 42):
    """
    Split PokemonData (ImageFolder) into train / val / test.
    Train split uses augmentation; val/test use resize-only transforms.
    """
    base = datasets.ImageFolder(data_dir)   # no transform, used only for split indices
    n       = len(base)
    n_val   = int(n * val_ratio)
    n_test  = int(n * test_ratio)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        base, [n_train, n_val, n_test], generator=gen
    )

    # Attach correct transforms (same file list, different augmentation)
    train_data = datasets.ImageFolder(data_dir, transform=get_transforms(True))
    eval_data  = datasets.ImageFolder(data_dir, transform=get_transforms(False))

    train_ds = Subset(train_data, train_sub.indices)
    val_ds   = Subset(eval_data,  val_sub.indices)
    test_ds  = Subset(eval_data,  test_sub.indices)

    return train_ds, val_ds, test_ds, base.classes


# --------------------------------------------------------------------------- #
# Model factory
# Lecture ref: AlexNet -> VGGNet -> GoogLeNet -> ResNet -> EfficientNet
# Key idea: replace classification head (FC-1000) with FC-num_classes
# --------------------------------------------------------------------------- #
def build_model(backbone: str, num_classes: int, pretrained: bool,
                freeze: bool, device: torch.device) -> nn.Module:
    """Build backbone and replace head with num_classes output."""
    if backbone == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model   = models.resnet34(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model   = models.resnet50(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model   = models.vgg16(weights=weights)
        if freeze:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif backbone == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model   = models.efficientnet_b0(weights=weights)
        if freeze:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError(f'Unknown backbone: {backbone}')

    return model.to(device)


# --------------------------------------------------------------------------- #
# Training / evaluation helpers
# Lecture ref: image_classification.py - model.eval(), torch.no_grad()
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n         = len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, correct / n, precision, recall, f1


# --------------------------------------------------------------------------- #
# Learning curve plot
# --------------------------------------------------------------------------- #
def _save_learning_curve(history: dict, exp_name: str, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='train', color='royalblue')
    ax1.plot(epochs, history['val_loss'],   label='val',   color='tomato')
    ax1.set_title(f'{exp_name} - Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='train', color='royalblue')
    ax2.plot(epochs, history['val_acc'],   label='val',   color='tomato')
    ax2.set_title(f'{exp_name} - Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(EXPERIMENTS[exp_name]['description'], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Run one experiment
# --------------------------------------------------------------------------- #
def run_experiment(exp_name: str, cfg: dict, data_dir: str,
                   results_dir: Path, device: torch.device) -> dict:
    print(f'\n{"="*65}')
    print(f'  {exp_name}: {cfg["description"]}')
    print(f'{"="*65}')

    out_dir = results_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────── #
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir)
    num_classes = len(class_names)
    with open(results_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f)
    print(f'  Classes: {num_classes} | '
          f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────── #
    model     = build_model(cfg['backbone'], num_classes,
                            cfg['pretrained'], cfg['freeze'], device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total_p:,} total | {trainable:,} trainable')

    # ── Optimizer & scheduler ──────────────────────────────────────────────── #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg['lr']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs']
    )

    # ── Training loop ─────────────────────────────────────────────────────── #
    history      = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, cfg['epochs'] + 1):
        tr_loss, tr_acc     = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, *_ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save({
                'state_dict':  model.state_dict(),
                'class_names': class_names,
                'backbone':    cfg['backbone'],
                'num_classes': num_classes,
            }, out_dir / 'best_model.pth')

        print(f'  [{epoch:3d}/{cfg["epochs"]}] '
              f'train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
              f'val loss={vl_loss:.4f} acc={vl_acc:.4f}')

    print(f'  => Best val_acc={best_val_acc:.4f} at epoch {best_epoch}')

    # ── Test evaluation ───────────────────────────────────────────────────── #
    ckpt = torch.load(out_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    _, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    print(f'  Test  acc={test_acc:.4f} | prec={test_prec:.4f} | '
          f'rec={test_rec:.4f} | f1={test_f1:.4f}')

    # ── Save ──────────────────────────────────────────────────────────────── #
    metrics = {
        'exp':             exp_name,
        'description':     cfg['description'],
        'test_acc':        round(test_acc,   4),
        'test_precision':  round(test_prec,  4),
        'test_recall':     round(test_rec,   4),
        'test_f1':         round(test_f1,    4),
        'best_val_acc':    round(best_val_acc, 4),
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    _save_learning_curve(history, exp_name, out_dir)
    print(f'  Saved -> {out_dir}')
    return metrics


# --------------------------------------------------------------------------- #
# Comparison chart across all experiments
# --------------------------------------------------------------------------- #
def plot_comparison(all_metrics: list, results_dir: Path):
    names  = [m['exp'] for m in all_metrics]
    keys   = ['test_acc', 'test_precision', 'test_recall', 'test_f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['royalblue', 'seagreen', 'tomato', 'darkorange']
    width  = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(names))
    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals   = [m[key] for m in all_metrics]
        offset = (i - 1.5) * width
        bars   = ax.bar([xi + offset for xi in x], vals, width,
                        label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Experiment Comparison - Test Set Performance')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nComparison chart saved -> {results_dir / "comparison.png"}')


# --------------------------------------------------------------------------- #
# Dataset path resolver (supports both manual path and kagglehub auto-download)
# --------------------------------------------------------------------------- #
def resolve_data_dir(data_dir: str | None, use_kagglehub: bool) -> str:
    """
    Return the path that contains Pokemon class sub-folders.

    kagglehub downloads to a cache dir like:
      ~/.cache/kaggle/datasets/lantian773030/pokemonclassification/versions/N/
    The actual ImageFolder-compatible root is inside that at 'PokemonData/'.
    We check both the returned path and '<path>/PokemonData/' automatically.
    """
    if use_kagglehub:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit(
                'kagglehub is not installed. Run: pip install kagglehub'
            )
        print('Downloading dataset via kagglehub ...')
        path = kagglehub.dataset_download('lantian773030/pokemonclassification')
        print(f'Downloaded to: {path}')
        # The dataset folder inside the cache is usually 'PokemonData/'
        candidate = Path(path) / 'PokemonData'
        if candidate.is_dir():
            return str(candidate)
        return path   # fallback: classes are directly inside path

    if data_dir is None:
        raise SystemExit(
            'Provide --data_dir <path> or use --kagglehub to auto-download.'
        )
    return data_dir


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description='Pokemon Classifier - Transfer Learning HW6',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {k}: {v["description"]}'
                         for k, v in EXPERIMENTS.items()),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_dir',   metavar='PATH',
                       help='Path to PokemonData directory (ImageFolder layout)')
    group.add_argument('--kagglehub',  action='store_true',
                       help='Auto-download dataset via kagglehub (no manual path needed)')
    parser.add_argument('--exp',         choices=list(EXPERIMENTS.keys()),
                        help='Run a single experiment')
    parser.add_argument('--all',         action='store_true',
                        help='Run all 4 experiments and generate comparison chart')
    parser.add_argument('--results_dir', default='results',
                        help='Output directory (default: ./results)')
    args = parser.parse_args()

    if not args.exp and not args.all:
        parser.error('Specify --exp <name> or --all')

    data_dir    = resolve_data_dir(args.data_dir, args.kagglehub)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device   : {device}')
    print(f'Data dir : {data_dir}')

    exps_to_run = list(EXPERIMENTS.keys()) if args.all else [args.exp]
    all_metrics = []

    for exp_name in exps_to_run:
        m = run_experiment(
            exp_name, EXPERIMENTS[exp_name],
            data_dir, results_dir, device,
        )
        all_metrics.append(m)

    if args.all and len(all_metrics) > 1:
        plot_comparison(all_metrics, results_dir)

    # Summary table
    print('\n' + '=' * 70)
    print(f'{"Experiment":<12} {"Acc":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}')
    print('-' * 70)
    for m in all_metrics:
        print(f'{m["exp"]:<12} '
              f'{m["test_acc"]:>8.4f} '
              f'{m["test_precision"]:>8.4f} '
              f'{m["test_recall"]:>8.4f} '
              f'{m["test_f1"]:>8.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
