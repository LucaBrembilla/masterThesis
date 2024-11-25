#!/bin/bash
if [ ! -f "./pretrained/epoch_160.pth" ]; then
  echo "Checkpoint not found: ./pretrained/epoch_160.pth"
  exit 1
fi

if [ ! -f "./dataset/demo_data/test/000002.bin" ]; then
  echo "Point cloud file not found: ./dataset/demo_data/test/000002.bin"
  exit 1
fi

python custom_test.py --ckpt ./pretrained/epoch_160.pth --pc_path ./dataset/demo_data/test/000002.bin
python custom_test.py --ckpt ./pretrained/epoch_160.pth --pc_path ./dataset/demo_data/val/000134.bin