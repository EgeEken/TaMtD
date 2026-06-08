import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from jpeg_decoder import JPEGDecoderModel
from jpeg_reader import JPEGReader


class HuggingFaceJPEGDataset(Dataset):
    def __init__(self, dataset_name, split, image_size, grayscale, quality, max_bytes, limit=None, image_column=None):
        self.dataset = load_dataset(dataset_name, split=split)
        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.image_column = image_column or self._find_image_column()
        self.reader = JPEGReader(
            image_size=image_size,
            grayscale=grayscale,
            quality=quality,
            max_bytes=max_bytes,
        )

    def _find_image_column(self):
        for name in ("image", "img"):
            if name in self.dataset.column_names:
                return name
        return self.dataset.column_names[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index][self.image_column]
        if not isinstance(item, Image.Image):
            item = Image.open(item)
        syntax, target = self.reader.encode_image(item)
        return syntax.bytes, syntax.markers, target


def psnr(pred, target):
    mse = F.mse_loss(pred, target).clamp_min(1e-8)
    return 10.0 * torch.log10(1.0 / mse)


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target) + F.mse_loss(pred, target) + 0.25 * gradient_loss(pred, target)


def save_preview(pred, target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred[0].detach().cpu().clamp(0, 1)
    target = target[0].detach().cpu().clamp(0, 1)
    image = torch.cat([target, pred], dim=2)
    if image.size(0) == 1:
        array = (image.squeeze(0).numpy() * 255).astype("uint8")
        Image.fromarray(array, mode="L").save(path)
    else:
        array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(array).save(path)


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    train_data = HuggingFaceJPEGDataset(
        args.dataset,
        args.train_split,
        args.image_size,
        args.grayscale,
        args.quality,
        args.max_bytes,
        args.train_limit,
        args.image_column,
    )
    val_data = HuggingFaceJPEGDataset(
        args.dataset,
        args.val_split,
        args.image_size,
        args.grayscale,
        args.quality,
        args.max_bytes,
        args.val_limit,
        args.image_column,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = JPEGDecoderModel(
        image_size=args.image_size,
        max_bytes=args.max_bytes,
        patch_size=args.patch_size,
        image_patch_size=args.image_patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        grayscale=args.grayscale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for step, (bytes_, markers, target) in enumerate(train_loader, 1):
            bytes_ = bytes_.to(device)
            markers = markers.to(device)
            target = target.to(device)
            pred = model(bytes_, markers)
            loss = reconstruction_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} train_loss={train_loss / step:.4f}")

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for step, (bytes_, markers, target) in enumerate(val_loader, 1):
                bytes_ = bytes_.to(device)
                markers = markers.to(device)
                target = target.to(device)
                pred = model(bytes_, markers)
                val_loss += reconstruction_loss(pred, target).item()
                val_psnr += psnr(pred, target).item()
                if step == 1:
                    save_preview(pred, target, output_dir / f"preview_epoch_{epoch}.png")

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_psnr /= max(len(val_loader), 1)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_psnr={val_psnr:.2f}")

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--output-dir", default="runs/jpeg_decoder_tiny")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--max-bytes", type=int, default=4096)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
