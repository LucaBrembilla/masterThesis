#!/usr/bin/env python3
"""
Evaluation script for 3D bounding boxes.

This script computes common detection statistics:
  - Per-class Average Precision (AP)
  - Mean Average Precision (mAP)
  - Precision, Recall, and F1-score
  - Counts of ground-truth boxes, predictions, TP, FP, FN
  - Average IoU for true positive detections

It supports single file inputs or folder inputs. In folder mode the files are
paired (by sorted order) between the prediction (.txt) files and the ground-truth (.json) files.
"""

import os
import glob
import json
import math
import argparse
import numpy as np
from shapely.geometry import Polygon

# -----------------------------------------------------------------------------
# Box3D class to hold box parameters
# -----------------------------------------------------------------------------
class Box3D:
    def __init__(self, class_name, x, y, z, l, w, h, yaw, confidence=None):
        self.class_name = class_name  # e.g. "CAR", "VAN", "TRAILER"
        self.x = x
        self.y = y
        self.z = z
        self.l = l  # length
        self.w = w  # width
        self.h = h  # height
        self.yaw = yaw  # rotation (in radians)
        self.confidence = confidence  # detection confidence (None for GT)
        self.matched = False  # flag used during matching

# -----------------------------------------------------------------------------
# Compute the 2D corners of a rotated rectangle in the XY plane
# -----------------------------------------------------------------------------
def get_2d_corners(box):
    """
    Compute the four corners of the box in the XY plane.
    The box is assumed to be centered at (x,y) with dimensions l and w and rotated by yaw.
    """
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    l2 = box.l / 2.0
    w2 = box.w / 2.0
    # Corners relative to center
    corners = np.array([
        [ l2,  w2],
        [ l2, -w2],
        [-l2, -w2],
        [-l2,  w2]
    ])
    # Rotate and translate
    R = np.array([[c, -s],
                  [s,  c]])
    rotated = np.dot(corners, R.T)
    rotated[:, 0] += box.x
    rotated[:, 1] += box.y
    return rotated

# -----------------------------------------------------------------------------
# Compute the 3D IoU between two boxes.
# -----------------------------------------------------------------------------
def compute_3d_iou(box1, box2):
    """
    Compute the 3D Intersection over Union (IoU) between two 3D boxes.
    Uses the 2D overlap in the bird's-eye view and the overlap in height.
    """
    poly1 = Polygon(get_2d_corners(box1))
    poly2 = Polygon(get_2d_corners(box2))
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    if inter_area == 0:
        return 0.0

    # Compute vertical (z-axis) overlap
    box1_zmin = box1.z - box1.h / 2.0
    box1_zmax = box1.z + box1.h / 2.0
    box2_zmin = box2.z - box2.h / 2.0
    box2_zmax = box2.z + box2.h / 2.0
    inter_z = max(0, min(box1_zmax, box2_zmax) - max(box1_zmin, box2_zmin))
    inter_vol = inter_area * inter_z
    vol1 = box1.l * box1.w * box1.h
    vol2 = box2.l * box2.w * box2.h
    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)
    return iou

# -----------------------------------------------------------------------------
# Compute AP given precision and recall arrays (VOC-style)
# -----------------------------------------------------------------------------
def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP) using the VOC method.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

# -----------------------------------------------------------------------------
# Load predictions from a text file
# -----------------------------------------------------------------------------
def load_predictions(file_path):
    """
    Load predicted boxes from a text file.
    Each non-comment line should have 9 numbers:
      class x y z l w h yaw confidence
    """
    boxes = []
    # Define your mapping (example: 1 -> "CAR", 2 -> "VAN", 3 -> "TRAILER")
    class_mapping = {1: "CAR", 2: "VAN", 3: "TRAILER"}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_int = int(parts[0])
            cls = class_mapping.get(cls_int, str(cls_int))
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            l = float(parts[4])
            w = float(parts[5])
            h = float(parts[6])
            yaw = float(parts[7])
            conf = float(parts[8])
            boxes.append(Box3D(cls, x, y, z, l, w, h, yaw, conf))
    return boxes

# -----------------------------------------------------------------------------
# Load ground truth boxes from a JSON file
# -----------------------------------------------------------------------------
def load_ground_truth(file_path):
    """
    Load ground truth boxes from a JSON file.
    """
    boxes = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for label in data.get("labels", []):
            cls = label.get("category", "UNKNOWN")
            box3d = label.get("box3d", {})
            location = box3d.get("location", {})
            dimension = box3d.get("dimension", {})
            orientation = box3d.get("orientation", {})
            x = location.get("x", 0.0)
            y = location.get("y", 0.0)
            z = location.get("z", 0.0)
            l = dimension.get("length", 0.0)
            w = dimension.get("width", 0.0)
            h = dimension.get("height", 0.0)
            yaw = orientation.get("rotationYaw", 0.0)
            boxes.append(Box3D(cls, x, y, z, l, w, h, yaw))
    return boxes

# -----------------------------------------------------------------------------
# Evaluate a dataset given lists of matching files (per image evaluation)
# -----------------------------------------------------------------------------
def evaluate_folder(pred_folder, gt_folder, iou_threshold=0.5):
    """
    Evaluate detections over a dataset given prediction and ground truth folders.
    Files are paired by sorted order.
    Returns:
       - results: a dict with per-class statistics
       - overall: a dict with aggregated statistics over all classes
    """
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.txt")))
    gt_files   = sorted(glob.glob(os.path.join(gt_folder, "*.json")))
    if not pred_files or not gt_files:
        raise ValueError("No prediction (.txt) or ground truth (.json) files found in the provided folders.")

    if len(pred_files) != len(gt_files):
        print("Warning: Number of prediction and ground-truth files differ. "
              "Using min(n_pred, n_gt) = {} pairs.".format(min(len(pred_files), len(gt_files))))
    n_files = min(len(pred_files), len(gt_files))

    # Containers for aggregated stats
    gt_count = {}               # per-class count of GT boxes
    predictions_by_class = {}   # per-class list of tuples: (confidence, tp, iou)

    # Process each paired file (each image)
    for i in range(n_files):
        pred_boxes = load_predictions(pred_files[i])
        gt_boxes = load_ground_truth(gt_files[i])
        # Get union of classes in this image
        classes = set([b.class_name for b in gt_boxes] + [b.class_name for b in pred_boxes])
        for cls in classes:
            # Collect GT and predicted boxes for this class in the image
            gt_cls = [b for b in gt_boxes if b.class_name == cls]
            pred_cls = [b for b in pred_boxes if b.class_name == cls]
            pred_cls.sort(key=lambda b: b.confidence if b.confidence is not None else 0, reverse=True)
            # Update ground truth count per class
            gt_count[cls] = gt_count.get(cls, 0) + len(gt_cls)
            # Mark all GT boxes as unmatched
            for g in gt_cls:
                g.matched = False
            # Process each prediction: match with the best available GT box
            for p in pred_cls:
                best_iou = 0.0
                best_gt = None
                for g in gt_cls:
                    iou = compute_3d_iou(p, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
                if best_iou >= iou_threshold and best_gt is not None and not best_gt.matched:
                    tp_flag = 1
                    best_gt.matched = True
                    matched_iou = best_iou
                else:
                    tp_flag = 0
                    matched_iou = 0.0
                if cls not in predictions_by_class:
                    predictions_by_class[cls] = []
                predictions_by_class[cls].append((p.confidence, tp_flag, matched_iou))

    # Now, compute per-class statistics
    results = {}
    for cls, preds in predictions_by_class.items():
        preds.sort(key=lambda x: x[0], reverse=True)
        confidences = np.array([p[0] for p in preds])
        tp_array = np.array([p[1] for p in preds])
        pred_count = len(preds)
        tp_cum = np.cumsum(tp_array)
        fp_array = 1 - tp_array
        fp_cum = np.cumsum(fp_array)
        precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall_arr = tp_cum / (gt_count.get(cls, 1) + 1e-6)
        ap = compute_ap(recall_arr, precision_arr)
        final_precision = precision_arr[-1] if len(precision_arr) > 0 else 0.0
        final_recall = recall_arr[-1] if len(recall_arr) > 0 else 0.0
        if final_precision + final_recall > 0:
            f1 = 2 * final_precision * final_recall / (final_precision + final_recall)
        else:
            f1 = 0.0
        ious = [p[2] for p in preds if p[1] == 1]
        avg_iou = np.mean(ious) if ious else 0.0
        TP = int(tp_cum[-1]) if len(tp_cum) > 0 else 0
        FP = pred_count - TP
        FN = gt_count.get(cls, 0) - TP
        results[cls] = {
            'AP': ap,
            'precision': final_precision,
            'recall': final_recall,
            'F1': f1,
            'gt_count': gt_count.get(cls, 0),
            'pred_count': pred_count,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'avg_iou': avg_iou,
            'precision_curve': precision_arr,
            'recall_curve': recall_arr
        }

    # Compute overall (micro-averaged) statistics
    total_TP = sum([results[cls]['TP'] for cls in results])
    total_FP = sum([results[cls]['FP'] for cls in results])
    total_gt = sum([results[cls]['gt_count'] for cls in results])
    overall_precision = total_TP / (total_TP + total_FP + 1e-6)
    overall_recall = total_TP / (total_gt + 1e-6)
    if overall_precision + overall_recall > 0:
        overall_F1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    else:
        overall_F1 = 0.0
    mAP = np.mean([results[cls]['AP'] for cls in results]) if results else 0.0
    overall = {
        'mAP': mAP,
        'precision': overall_precision,
        'recall': overall_recall,
        'F1': overall_F1,
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_gt - total_TP,
        'gt_count': total_gt,
        'pred_count': sum([results[cls]['pred_count'] for cls in results])
    }
    return results, overall

# -----------------------------------------------------------------------------
# Main: determine file or folder mode and run evaluation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics (mAP, precision, recall, F1, etc.) for 3D bounding box detections."
    )
    parser.add_argument('--pred', required=True, help='Path to prediction file or folder')
    parser.add_argument('--gt', required=True, help='Path to ground truth file or folder')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching (default: 0.5)')
    args = parser.parse_args()

    if os.path.isdir(args.pred) and os.path.isdir(args.gt):
        print("Evaluating folders...")
        per_class_results, overall_results = evaluate_folder(args.pred, args.gt, iou_threshold=args.iou)
        print("----- Per-Class Evaluation -----")
        for cls, stats in per_class_results.items():
            print(f"Class: {cls}")
            print(f"  GT boxes:       {stats['gt_count']}")
            print(f"  Predicted boxes:{stats['pred_count']}")
            print(f"  TP: {stats['TP']}  FP: {stats['FP']}  FN: {stats['FN']}")
            print(f"  Precision: {stats['precision']:.4f}")
            print(f"  Recall:    {stats['recall']:.4f}")
            print(f"  F1-score:  {stats['F1']:.4f}")
            print(f"  AP:        {stats['AP']:.4f}")
            print(f"  Avg IoU (TP): {stats['avg_iou']:.4f}")
            print("")
        print("----- Overall Evaluation -----")
        print(f"Total GT boxes:    {overall_results['gt_count']}")
        print(f"Total predictions: {overall_results['pred_count']}")
        print(f"TP: {overall_results['TP']}  FP: {overall_results['FP']}  FN: {overall_results['FN']}")
        print(f"Precision: {overall_results['precision']:.4f}")
        print(f"Recall:    {overall_results['recall']:.4f}")
        print(f"F1-score:  {overall_results['F1']:.4f}")
        print(f"mAP:       {overall_results['mAP']:.4f}")
    elif os.path.isfile(args.pred) and os.path.isfile(args.gt):
        print("Evaluating single files...")
        # For single file evaluation, we simply load and evaluate one image.
        pred_boxes = load_predictions(args.pred)
        gt_boxes = load_ground_truth(args.gt)
        # We use a similar matching procedure as in folder mode.
        classes = set([b.class_name for b in gt_boxes] + [b.class_name for b in pred_boxes])
        per_class_results = {}
        for cls in classes:
            gt_cls = [b for b in gt_boxes if b.class_name == cls]
            pred_cls = [b for b in pred_boxes if b.class_name == cls]
            pred_cls.sort(key=lambda b: b.confidence if b.confidence is not None else 0, reverse=True)
            for g in gt_cls:
                g.matched = False
            tp_list = []
            iou_list = []
            for p in pred_cls:
                best_iou = 0.0
                best_gt = None
                for g in gt_cls:
                    iou = compute_3d_iou(p, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
                if best_iou >= args.iou and best_gt is not None and not best_gt.matched:
                    tp_list.append(1)
                    best_gt.matched = True
                    iou_list.append(best_iou)
                else:
                    tp_list.append(0)
            tp_array = np.array(tp_list)
            pred_count = len(pred_cls)
            tp_cum = np.cumsum(tp_array)
            fp_array = 1 - tp_array
            fp_cum = np.cumsum(fp_array)
            precision_arr = tp_cum / (tp_cum + fp_cum + 1e-6)
            recall_arr = tp_cum / (len(gt_cls) + 1e-6)
            ap = compute_ap(recall_arr, precision_arr) if pred_count > 0 else 0.0
            final_precision = precision_arr[-1] if len(precision_arr) > 0 else 0.0
            final_recall = recall_arr[-1] if len(recall_arr) > 0 else 0.0
            if final_precision + final_recall > 0:
                f1 = 2 * final_precision * final_recall / (final_precision + final_recall)
            else:
                f1 = 0.0
            avg_iou = np.mean(iou_list) if iou_list else 0.0
            per_class_results[cls] = {
                'AP': ap,
                'precision': final_precision,
                'recall': final_recall,
                'F1': f1,
                'gt_count': len(gt_cls),
                'pred_count': pred_count,
                'TP': int(tp_cum[-1]) if len(tp_cum) > 0 else 0,
                'FP': pred_count - (int(tp_cum[-1]) if len(tp_cum) > 0 else 0),
                'FN': len(gt_cls) - (int(tp_cum[-1]) if len(tp_cum) > 0 else 0),
                'avg_iou': avg_iou,
                'precision_curve': precision_arr,
                'recall_curve': recall_arr
            }
        print("----- Per-Class Evaluation -----")
        for cls, stats in per_class_results.items():
            print(f"Class: {cls}")
            print(f"  GT boxes:       {stats['gt_count']}")
            print(f"  Predicted boxes:{stats['pred_count']}")
            print(f"  TP: {stats['TP']}  FP: {stats['FP']}  FN: {stats['FN']}")
            print(f"  Precision: {stats['precision']:.4f}")
            print(f"  Recall:    {stats['recall']:.4f}")
            print(f"  F1-score:  {stats['F1']:.4f}")
            print(f"  AP:        {stats['AP']:.4f}")
            print(f"  Avg IoU (TP): {stats['avg_iou']:.4f}")
            print("")
    else:
        raise ValueError("Both --pred and --gt should be either files or folders.")

if __name__ == '__main__':
    main()
