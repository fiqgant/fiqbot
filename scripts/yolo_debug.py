#!/usr/bin/env python3
"""
YOLO Debug Script - Beautiful Edition
A professional debugging tool for YOLO object detection.

Features:
- Detects all 80 COCO classes with colored bounding boxes
- Real-time FPS and inference time display
- Detection statistics panel
- Screenshot capture (press 's')
- Toggle detection info (press 'i')
- Pause/Resume (press 'space')
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque

# =========================================================
# CONFIG
# =========================================================
ONNX_PATH = "yolo11n.onnx"
IMG_SIZE = 320

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

CONF_TH = 0.35
NMS_TH = 0.45

WINDOW_NAME = "YOLO Debug - Beautiful Edition"

# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Beautiful color palette (vibrant colors)
COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255),
    (100, 255, 255), (255, 180, 100), (180, 100, 255), (100, 255, 180), (255, 100, 180),
    (180, 255, 100), (100, 180, 255), (220, 220, 100), (220, 100, 220), (100, 220, 220),
]


def get_color(class_id):
    """Get a consistent color for each class"""
    return COLORS[class_id % len(COLORS)]


def create_ort_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]


def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = image.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((nh, nw, 3), color, dtype=np.uint8)
    pad_w = (nw - new_w) // 2
    pad_h = (nh - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def to_blob(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def parse_output_all(outs, conf_th):
    """Parse YOLO output and return ALL detections"""
    out = np.array(outs[0])
    if out.ndim == 3:
        out = out[0]
    
    # Standard YOLO11 output: [N, 6] where cols are [x1, y1, x2, y2, conf, class]
    if out.ndim == 2 and out.shape[1] >= 6:
        boxes = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]
    
    # Transposed format: [84, N] where 84 = 4 (bbox) + 80 (classes)
    if out.ndim == 2 and out.shape[0] == 84:
        out = out.T
        boxes = out[:, 0:4]
        class_probs = out[:, 4:]
        cls = np.argmax(class_probs, axis=1)
        scores = np.max(class_probs, axis=1)
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]
    
    return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int32)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    """Draw a rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw filled rounded rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_fancy_box(frame, x1, y1, x2, y2, color, label, score):
    """Draw a fancy bounding box with label"""
    # Semi-transparent overlay for label background
    overlay = frame.copy()
    
    # Draw main box with thicker lines
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw corner accents
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    thick = 3
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick)
    
    # Label background
    label_text = f"{label}: {score:.0%}"
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    label_y = y1 - 8 if y1 > 30 else y2 + th + 8
    
    cv2.rectangle(overlay, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 4), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Label text
    cv2.putText(frame, label_text, (x1 + 4, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_panel(frame, fps, infer_ms, detections, class_counts, show_details):
    """Draw info panel with detection statistics"""
    h, w = frame.shape[:2]
    
    # Semi-transparent black panel at top
    overlay = frame.copy()
    panel_h = 40 if not show_details else 40 + min(len(class_counts), 5) * 20 + 10
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Main stats
    stats_text = f"FPS: {fps:2d}  |  Infer: {infer_ms:.1f}ms  |  Objects: {detections}  |  Size: {IMG_SIZE}"
    cv2.putText(frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Help text
    help_text = "[Q]uit  [S]creenshot  [I]nfo  [SPACE]Pause"
    (tw, _), _ = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, help_text, (w - tw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Detailed class breakdown
    if show_details and class_counts:
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (cls_name, count) in enumerate(sorted_counts):
            y = 55 + i * 20
            text = f"  {cls_name}: {count}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_no_detection(frame):
    """Draw message when no detection"""
    h, w = frame.shape[:2]
    text = "No objects detected"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = (w - tw) // 2
    y = (h + th) // 2
    
    # Draw shadow
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)


def main():
    if not os.path.isfile(ONNX_PATH):
        print(f"[ERROR] Model not found: {ONNX_PATH}")
        print("Please ensure yolo11n.onnx is in the current directory")
        return
    
    print(f"╔══════════════════════════════════════════╗")
    print(f"║     YOLO Debug - Beautiful Edition       ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Model: {ONNX_PATH:<30} ║")
    print(f"║  Input Size: {IMG_SIZE}x{IMG_SIZE:<24} ║")
    print(f"║  Confidence: {CONF_TH:.0%}                        ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Controls:                               ║")
    print(f"║    Q     - Quit                          ║")
    print(f"║    S     - Save screenshot               ║")
    print(f"║    I     - Toggle info panel             ║")
    print(f"║    SPACE - Pause/Resume                  ║")
    print(f"╚══════════════════════════════════════════╝")
    
    sess, in_name, out_names = create_ort_session(ONNX_PATH)
    
    # Print model info
    inp = sess.get_inputs()[0]
    print(f"\n[INFO] Model input: {inp.name} shape={inp.shape}")
    
    print(f"[INFO] Opening camera {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("[INFO] Camera opened successfully!")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    
    # FPS tracking
    fps_deque = deque(maxlen=30)
    show_details = True
    paused = False
    last_frame = None
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    continue
                last_frame = frame.copy()
            else:
                if last_frame is None:
                    continue
                frame = last_frame.copy()
            
            h0, w0 = frame.shape[:2]
            
            # Preprocess
            img, scale, pad_w, pad_h = letterbox(frame, (IMG_SIZE, IMG_SIZE))
            blob = to_blob(img)
            
            # Inference
            t0 = time.time()
            outs = sess.run(out_names, {in_name: blob})
            infer_ms = (time.time() - t0) * 1000
            
            # Parse ALL detections
            boxes, scores, cls = parse_output_all(outs, CONF_TH)
            
            # NMS per class
            final_boxes = []
            final_scores = []
            final_cls = []
            
            for c in np.unique(cls):
                mask = cls == c
                c_boxes = boxes[mask]
                c_scores = scores[mask]
                
                if len(c_boxes) == 0:
                    continue
                
                nms_boxes = []
                for b in c_boxes:
                    x1, y1, x2, y2 = b
                    nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                
                idxs = cv2.dnn.NMSBoxes(nms_boxes, c_scores.tolist(), CONF_TH, NMS_TH)
                if len(idxs) > 0:
                    idxs = idxs.flatten()
                    for i in idxs:
                        final_boxes.append(c_boxes[i])
                        final_scores.append(c_scores[i])
                        final_cls.append(c)
            
            # Count classes
            class_counts = {}
            for c in final_cls:
                name = COCO_NAMES[c] if c < len(COCO_NAMES) else f"cls{c}"
                class_counts[name] = class_counts.get(name, 0) + 1
            
            # Draw detections
            if len(final_boxes) == 0:
                draw_no_detection(frame)
            else:
                for i, box in enumerate(final_boxes):
                    x1, y1, x2, y2 = box
                    
                    # Transform back to original coords
                    x1 = int((x1 - pad_w) / scale)
                    y1 = int((y1 - pad_h) / scale)
                    x2 = int((x2 - pad_w) / scale)
                    y2 = int((y2 - pad_h) / scale)
                    
                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w0, x2), min(h0, y2)
                    
                    c = final_cls[i]
                    color = get_color(c)
                    name = COCO_NAMES[c] if c < len(COCO_NAMES) else f"cls{c}"
                    
                    draw_fancy_box(frame, x1, y1, x2, y2, color, name, final_scores[i])
            
            # FPS calculation
            fps_deque.append(time.time())
            if len(fps_deque) >= 2:
                fps = int(len(fps_deque) / (fps_deque[-1] - fps_deque[0] + 0.001))
            else:
                fps = 0
            
            # Draw info panel
            draw_info_panel(frame, fps, infer_ms, len(final_boxes), class_counts, show_details)
            
            # Pause indicator
            if paused:
                cv2.putText(frame, "PAUSED", (w0 // 2 - 60, h0 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                filename = f"yolo_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved: {filename}")
            elif key == ord('i'):
                show_details = not show_details
            elif key == ord(' '):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
