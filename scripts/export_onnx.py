"""
Export a trained PyTorch model checkpoint to ONNX.

Defaults:
- Loads checkpoint from models/best_model.pth
- Loads labels from models/label_map.json (fallback to defaults)
- Builds ResNet50 backbone with a Linear head sized to the number of labels
- Exports ONNX to mobile_exports/model.onnx

Usage examples:
  python scripts/export_onnx.py \
	--checkpoint models/best_model.pth \
	--labels models/label_map.json \
	--output mobile_exports/model.onnx \
	--opset 17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import models

DEFAULT_LABELS = [
	"angry",
	"fearful",
	"happy",
	"neutral",
	"sad",
	"surprised",
]


def load_labels(labels_path: Optional[Path | str]) -> List[str]:
	if not labels_path:
		return DEFAULT_LABELS
	p = Path(labels_path)
	if not p.exists():
		return DEFAULT_LABELS
	try:
		data = json.loads(p.read_text())
		# label_map format is {label: index}; sort by index
		if isinstance(data, dict):
			items = sorted(data.items(), key=lambda kv: kv[1])
			return [k for k, _ in items]
		if isinstance(data, list) and all(isinstance(x, str) for x in data):
			return data
	except Exception:
		pass
	return DEFAULT_LABELS


def build_model(num_classes: int) -> nn.Module:
	backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
	in_features = backbone.fc.in_features
	backbone.fc = nn.Linear(in_features, num_classes)
	return backbone


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
	try:
		state = torch.load(ckpt_path, map_location="cpu")
	except Exception:
		state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]

	if isinstance(state, dict) and "state_dict" in state:
		model.load_state_dict(state["state_dict"], strict=False)
		return
	if isinstance(state, dict) and "model_state_dict" in state:
		model.load_state_dict(state["model_state_dict"], strict=False)
		return
	if isinstance(state, dict) and all(
		k.startswith("features")
		or k.startswith("classifier")
		or k.startswith("fc.")
		or k.startswith("layer")
		for k in state.keys()
	):
		model.load_state_dict(state, strict=False)
		return
	# Maybe an entire model object
	if hasattr(state, "state_dict"):
		model.load_state_dict(state.state_dict(), strict=False)  # type: ignore[attr-defined]
		return
	raise ValueError("Unsupported checkpoint format for ONNX export")


def export_onnx(
	checkpoint: Path,
	labels_path: Optional[Path | str],
	output_path: Path,
	opset: int = 17,
) -> None:
	labels = load_labels(labels_path)
	num_classes = len(labels)
	model = build_model(num_classes)
	load_checkpoint(model, checkpoint)
	model.eval()

	# Dummy input (N, C, H, W); model expects 224x224 RGB with ImageNet normalization upstream.
	dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

	dynamic_axes = {
		"input": {0: "batch"},
		"logits": {0: "batch"},
	}

	output_path.parent.mkdir(parents=True, exist_ok=True)

	torch.onnx.export(
		model,
		dummy,
		str(output_path),
		input_names=["input"],
		output_names=["logits"],
		dynamic_axes=dynamic_axes,
		opset_version=opset,
		do_constant_folding=True,
	)

	print(f"Exported ONNX to {output_path} (opset={opset}, classes={num_classes})")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export PyTorch model checkpoint to ONNX")
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=Path("models/best_model.pth"),
		help="Path to the .pth checkpoint",
	)
	parser.add_argument(
		"--labels",
		type=Path,
		default=Path("models/label_map.json"),
		help="Path to labels json or list file",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("mobile_exports/model.onnx"),
		help="Destination ONNX file path",
	)
	parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	export_onnx(
		checkpoint=args.checkpoint,
		labels_path=args.labels,
		output_path=args.output,
		opset=args.opset,
	)
