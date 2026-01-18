#!/usr/bin/env python3
"""
YOLO Debug Script - Detects ALL objects with bounding boxes
Run this to verify YOLO detection is working properly.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import onnxruntime as ort
import time

# =========================================================
# CONFIG
# =========================================================
ONNX_PATH = "yolo11n.onnx"
IMG_SIZE = 320

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

CONF_TH = 0.25  # Confidence threshold
NMS_TH = 0.45

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

# Random colors for each class
np.random.seed(42)
COLORS = [(int(r), int(g), int(b)) for r, g, b in np.random.randint(0, 255, (80, 3))]


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
    """Parse YOLO output and return ALL detections (not just person)"""
    out = np.array(outs[0])
    if out.ndim == 3:
        out = out[0]
    
    print(f"[DEBUG] Raw output shape: {out.shape}")
    
    # Standard YOLO11 output: [N, 6] where cols are [x1, y1, x2, y2, conf, class]
    if out.ndim == 2 and out.shape[1] >= 6:
        boxes = out[:, 0:4].astype(np.float32)
        scores = out[:, 4].astype(np.float32)
        cls = out[:, 5].astype(np.int32)
        
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]
    
    # Transposed format: [84, N] where 84 = 4 (bbox) + 80 (classes)
    if out.ndim == 2 and out.shape[0] == 84:
        out = out.T  # Now [N, 84]
        boxes = out[:, 0:4]
        class_probs = out[:, 4:]  # 80 classes
        
        # Get best class for each detection
        cls = np.argmax(class_probs, axis=1)
        scores = np.max(class_probs, axis=1)
        
        keep = scores > conf_th
        return boxes[keep], scores[keep], cls[keep]
    
    print(f"[WARNING] Unknown output format: {out.shape}")
    return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int32)


def main():
    if not os.path.isfile(ONNX_PATH):
        print(f"[ERROR] Model not found: {ONNX_PATH}")
        print("Please ensure yolo11n.onnx is in the current directory")
        return
    
    print(f"Loading model: {ONNX_PATH}")
    sess, in_name, out_names = create_ort_session(ONNX_PATH)
    
    # Print model info
    inp = sess.get_inputs()[0]
    print(f"Model input: {inp.name} shape={inp.shape}")
    print(f"Using IMG_SIZE: {IMG_SIZE}")
    
    print(f"Opening camera {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
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
            
            # Convert to xywh for NMS
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
        
        # Draw detections
        for i, box in enumerate(final_boxes):
            x1, y1, x2, y2 = box
            
            # Transform back to original coords
            x1 = int((x1 - pad_w) / scale)
            y1 = int((y1 - pad_h) / scale)
            x2 = int((x2 - pad_w) / scale)
            y2 = int((y2 - pad_h) / scale)
            
            # Clamp
            x1 = max(0, min(x1, w0))
            y1 = max(0, min(y1, h0))
            x2 = max(0, min(x2, w0))
            y2 = max(0, min(y2, h0))
            
            c = final_cls[i]
            score = final_scores[i]
            color = COLORS[c % len(COLORS)]
            name = COCO_NAMES[c] if c < len(COCO_NAMES) else f"cls{c}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS calculation
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        # Info overlay
        info = f"FPS: {fps} | Infer: {infer_ms:.1f}ms | Detections: {len(final_boxes)} | Size: {IMG_SIZE}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("YOLO Debug - All Classes", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"debug_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
