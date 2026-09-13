

import argparse
import cv2
import numpy as np
import sys
import time
import colorsys
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 偵測後端：.hef → Hailo NPU（HailoRT）、.pt → ultralytics（CPU/GPU）
# 兩者都延遲載入，避免不需要時還得等 torch import。
# ============================================================
try:
    import hailo_platform  # noqa: F401
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False


# ============================================================
# 1. 卡爾曼濾波器（對應 SimpleKalman2D）
# ============================================================
class SimpleKalman2D:
    """
    2D 定速卡爾曼濾波器
    狀態向量: [x, y, vx, vy]
    """
    def __init__(self, x0: float, y0: float, init_vel: float = 0.0):
        self.X = np.array([x0, y0, init_vel, init_vel], dtype=np.float32)
        self.P = np.diag([50, 50, 100, 100]).astype(np.float32)
        self.Q = np.diag([1, 1, 10, 10]).astype(np.float32)
        self.R = np.diag([25, 25]).astype(np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self._vx = 0.0
        self._vy = 0.0
        self.alpha = 0.5  # 速度平滑係數

    def set_delta_t(self, dt: float):
        dt = max(1e-3, dt)
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        s = max(1e-3, dt)
        self.Q = np.diag([s, s, 10*s, 10*s]).astype(np.float32)

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx: float, zy: float):
        z = np.array([zx, zy], dtype=np.float32)
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P
        # 平滑速度
        new_vx = float(self.X[2])
        new_vy = float(self.X[3])
        self._vx = (1 - self.alpha) * self._vx + self.alpha * new_vx
        self._vy = (1 - self.alpha) * self._vy + self.alpha * new_vy

    @property
    def x(self): return float(self.X[0])
    @property
    def y(self): return float(self.X[1])
    @property
    def vx(self): return self._vx
    @property
    def vy(self): return self._vy


# ============================================================
# 2. 軌跡物件（對應 Track class）
# ============================================================
@dataclass
class Track:
    id: int
    kf: SimpleKalman2D
    last_update_ms: float
    miss: int = 0
    history: list = field(default_factory=list)
    MAX_HISTORY: int = 32

    @classmethod
    def create(cls, track_id: int, cx: float, cy: float, now_ms: float):
        kf = SimpleKalman2D(cx, cy)
        t = cls(id=track_id, kf=kf, last_update_ms=now_ms)
        t.history.append((cx, cy))
        return t

    def predict(self, now_ms: float):
        dt = max(0.001, (now_ms - self.last_update_ms) / 1000.0)
        self.kf.set_delta_t(dt)
        self.kf.predict()

    def update(self, cx: float, cy: float, now_ms: float):
        dt = max(0.001, (now_ms - self.last_update_ms) / 1000.0)
        self.kf.set_delta_t(dt)
        self.kf.update(cx, cy)
        self.last_update_ms = now_ms
        self.history.append((self.kf.x, self.kf.y))
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
        self.miss = 0

    def predicted_point(self, future_sec: float) -> tuple:
        px = self.kf.x + self.kf.vx * future_sec
        py = self.kf.y + self.kf.vy * future_sec
        return (px, py)


# ============================================================
# 3. 追蹤器（對應 MainActivity 裡的 track 邏輯）
# ============================================================
class Tracker:
    def __init__(self, assoc_dist_px: float = 160.0, max_miss: int = 4):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.assoc_dist_px = assoc_dist_px
        self.max_miss = max_miss

    def update(self, detections: list[tuple], now_ms: float) -> list[Track]:
        """
        detections: list of (cx, cy)
        回傳目前所有存活 Track
        """
        # 預測所有 track
        for t in self.tracks:
            t.predict(now_ms)

        matched_track_ids = set()
        matched_det_ids = set()

        # 貪婪匹配：最近鄰
        for di, (cx, cy) in enumerate(detections):
            best_dist = self.assoc_dist_px
            best_tid = None
            for t in self.tracks:
                if t.id in matched_track_ids:
                    continue
                dist = np.hypot(t.kf.x - cx, t.kf.y - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = t.id
            if best_tid is not None:
                for t in self.tracks:
                    if t.id == best_tid:
                        t.update(cx, cy, now_ms)
                        break
                matched_track_ids.add(best_tid)
                matched_det_ids.add(di)

        # 未匹配偵測 → 新 track
        for di, (cx, cy) in enumerate(detections):
            if di not in matched_det_ids:
                new_track = Track.create(self.next_id, cx, cy, now_ms)
                self.tracks.append(new_track)
                matched_track_ids.add(new_track.id)
                self.next_id += 1

        # 未匹配 track → miss++，超過限制刪除
        for t in self.tracks:
            if t.id not in matched_track_ids:
                t.miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        return self.tracks


# ============================================================
# 4. 覆蓋層繪製（對應 OverlayView.java onDraw）
# ============================================================
class OverlayRenderer:
    PREDICTION_HORIZON_SEC = 2.0

    def draw(self, frame: np.ndarray, tracks: list[Track], fps: float,
             alert: bool, monitor_region: tuple[int, int, int, int]):
        h, w = frame.shape[:2]

        # 畫警戒範圍：正常為黃色，觸發警報後為紅色。
        rx1, ry1, rx2, ry2 = monitor_region
        zone_color = (0, 0, 255) if alert else (0, 215, 255)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), zone_color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), zone_color, 3)
        cv2.putText(frame, "WARNING ZONE", (rx1 + 8, max(25, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)

        for t in tracks:
            color = self._track_color(t.id)

            # 畫軌跡線
            if len(t.history) >= 2:
                pts = np.array(t.history, dtype=np.int32)
                for i in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, 2)

            # 畫目前位置圓
            cx, cy = int(t.kf.x), int(t.kf.y)
            cv2.circle(frame, (cx, cy), 8, color, -1)
            cv2.putText(frame, f"ID{t.id}", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 畫預測點（虛線圓 → 用較小圓代替）
            px, py = t.predicted_point(self.PREDICTION_HORIZON_SEC)
            px, py = int(px), int(py)
            cv2.circle(frame, (px, py), 14, (255, 0, 255), 2)
            cv2.putText(frame, "pred", (px+8, py+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1)

        # 警報顯示
        if alert:
            self._draw_alert(frame, w)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 0), 2)

        return frame

    def _draw_alert(self, frame, w):
        # OpenCV 內建字型不支援中文，避免顯示成方塊。
        msg = "ALERT: OBJECT IN WARNING ZONE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(msg, font, font_scale, thickness)
        cx = w // 2
        x = cx - tw // 2
        y = 100
        pad = 12
        # 白色背景
        cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad),
                      (255,255,255), -1)
        cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad),
                      (0,0,255), 2)
        cv2.putText(frame, msg, (x, y), font, font_scale, (0,0,200), thickness)

    def _track_color(self, track_id: int) -> tuple:
        hue = (track_id * 47) % 360
        r, g, b = colorsys.hsv_to_rgb(hue/360.0, 0.9, 1.0)
        return (int(b*255), int(g*255), int(r*255))  # OpenCV BGR


# ============================================================
# 5. 偵測器包裝（YOLOv8 或 OpenCV DNN）
# ============================================================
# COCO 類別：只保留人與道路車輛。
TARGET_CLASS_IDS = (0, 1, 2, 3, 5, 7)
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
              5: "bus", 7: "truck"}


class YoloDetector:
    """ultralytics YOLOv8（CPU / GPU）。"""

    def __init__(self, model_path: str, conf: float):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        回傳 list of (cx, cy, label, score, x1, y1, x2, y2)
        """
        results = []
        # 在模型推論階段就排除其他類別，避免它們進入後續移動追蹤。
        res = self.model(
            frame,
            conf=self.conf,
            classes=list(TARGET_CLASS_IDS),
            verbose=False,
        )[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_id = int(box.cls)
            if class_id not in TARGET_CLASS_IDS:
                continue
            label = self.model.names[class_id]
            score = float(box.conf)
            results.append((cx, cy, label, score, x1, y1, x2, y2))
        return results

    def close(self):
        pass


class HailoDetector:
    """HailoRT 推論 .hef（NMS 已編進模型，輸出每類一個 (n, 5) 陣列）。

    輸出列格式：[y_min, x_min, y_max, x_max, score]，座標為 0~1 正規化。
    """

    def __init__(self, model_path: str, conf: float):
        from hailo_platform import (VDevice, HailoSchedulingAlgorithm,
                                    FormatType)
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.infer_model = self.vdevice.create_infer_model(model_path)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        self.conf = conf

        in_shape = self.infer_model.input().shape  # [H, W, C]
        self.in_h, self.in_w = in_shape[0], in_shape[1]

        self.configured = self.infer_model.configure()
        self.bindings = self.configured.create_bindings()
        self.out_buf = np.zeros(self.infer_model.output().shape, np.float32)
        self.bindings.output().set_buffer(self.out_buf)

    def detect(self, frame: np.ndarray) -> list[tuple]:
        h, w = frame.shape[:2]
        # 模型吃 RGB；picamera2 / OpenCV 給的是 BGR。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = np.ascontiguousarray(cv2.resize(rgb, (self.in_w, self.in_h)))
        self.bindings.input().set_buffer(inp)
        self.configured.run([self.bindings], 1000)

        results = []
        for class_id, boxes in enumerate(self.bindings.output().get_buffer()):
            if class_id not in TARGET_CLASS_IDS or len(boxes) == 0:
                continue
            label = COCO_NAMES[class_id]
            for ymin, xmin, ymax, xmax, score in boxes:
                if score < self.conf:
                    continue
                x1 = int(np.clip(xmin * w, 0, w - 1))
                y1 = int(np.clip(ymin * h, 0, h - 1))
                x2 = int(np.clip(xmax * w, 0, w - 1))
                y2 = int(np.clip(ymax * h, 0, h - 1))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                results.append((cx, cy, label, float(score), x1, y1, x2, y2))
        return results

    def close(self):
        # HailoRT 物件若交給直譯器結束時的 GC 亂序銷毀會 segfault，
        # 必須依 bindings → configured → infer_model → vdevice 順序明確釋放。
        del self.bindings, self.out_buf
        self.configured.shutdown()
        del self.configured, self.infer_model
        self.vdevice.release()


def Detector(model_path: str, conf: float = 0.4):
    """依副檔名選擇後端：.hef → Hailo，其餘交給 ultralytics。"""
    if model_path.lower().endswith(".hef"):
        if not HAS_HAILO:
            raise RuntimeError("找不到 hailo_platform，請確認 venv 有開 --system-site-packages")
        print(f"[INFO] 使用 Hailo NPU 推論：{model_path}")
        return HailoDetector(model_path, conf)
    print(f"[INFO] 使用 ultralytics 推論：{model_path}")
    return YoloDetector(model_path, conf)


# ============================================================
# 6. 主程式（對應 onCreate / onResume / camera loop）
# ============================================================
def check_in_monitor(px: float, py: float,
                     monitor_region: tuple[int, int, int, int]) -> bool:
    rx1, ry1, rx2, ry2 = monitor_region
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


class WarningZone:
    """可用滑鼠左鍵拖曳的警戒區。"""
    def __init__(self):
        self.region = (350, 200, 930, 650)
        self.start: Optional[tuple[int, int]] = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x1, y1 = self.start
            if abs(x - x1) >= 20 and abs(y - y1) >= 20:
                self.region = (min(x1, x), min(y1, y),
                               max(x1, x), max(y1, y))
                print(f"[INFO] 警戒區已更新：{self.region}")
            self.start = None


class AlertController:
    """只在進入區域的瞬間發聲，避免每幀連續響。"""
    def __init__(self, cooldown_sec: float = 1.5):
        self.was_alerting = False
        self.last_sound_time = 0.0
        self.cooldown_sec = cooldown_sec

    def update(self, alerting: bool):
        now = time.time()
        if alerting and not self.was_alerting and now - self.last_sound_time >= self.cooldown_sec:
            print("\a[ALERT] 物體進入警戒範圍！", flush=True)
            self.last_sound_time = now
        self.was_alerting = alerting


def list_macos_cameras() -> list[str]:
    """用 AVFoundation 列出相機名稱（順序與 OpenCV 的索引一致）。

    需要 pyobjc（pip install pyobjc-framework-AVFoundation）；沒有就回傳空清單。
    """
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation  # type: ignore
    except ImportError:
        return []

    try:
        device_types = [
            AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVFoundation.AVCaptureDeviceTypeExternal,
            AVFoundation.AVCaptureDeviceTypeContinuityCamera,
        ]
    except AttributeError:
        # 舊版 macOS / pyobjc 沒有 External 或 ContinuityCamera 常數
        device_types = [AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera]

    discovery = AVFoundation.AVCaptureDeviceDiscoverySession
    session = discovery.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types,
        AVFoundation.AVMediaTypeVideo,
        AVFoundation.AVCaptureDevicePositionUnspecified,
    )
    return [str(d.localizedName()) for d in session.devices()]


def find_phone_camera_index(names: Optional[list[str]] = None) -> Optional[int]:
    """在相機名稱清單中找出 iPhone / 接續互通相機的索引。"""
    if names is None:
        names = list_macos_cameras()
    keywords = ("iphone", "continuity", "ipad", "phone")
    for idx, name in enumerate(names):
        if any(k in name.lower() for k in keywords):
            print(f"[INFO] 找到手機相機：index {idx} - {name}")
            return idx
    if names:
        print(f"[WARN] 目前沒有手機相機可用，現有裝置：{names}")
        print("       請讓 iPhone 解鎖並靠近 Mac，確認已開啟「接續互通相機」。")
    return None


class PiCamera:
    """用 picamera2 讀取 Raspberry Pi CSI 相機，介面模仿 cv2.VideoCapture。

    Pi 5 上 CSI 相機的 /dev/video0 只吐 raw Bayer，cv2.VideoCapture 拿不到可用畫面。
    """
    def __init__(self, width: int = 1280, height: int = 720):
        from picamera2 import Picamera2  # type: ignore
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        frame = self.cam.capture_array("main")
        if frame is None:
            return False, None
        # picamera2 的 RGB888 實際記憶體排列與 OpenCV 的 BGR 相同，直接使用即可。
        return True, frame

    def set(self, *_args):
        # 解析度已在建構時設定，忽略後續 cap.set() 呼叫。
        pass

    def release(self):
        self.cam.stop()
        self.cam.close()


def open_pi_camera() -> Optional[PiCamera]:
    """有 picamera2 且有 CSI 相機時回傳 PiCamera，否則回傳 None。"""
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError:
        return None
    try:
        if not Picamera2.global_camera_info():
            return None
        cam = PiCamera()
        ok, _ = cam.read()
        if ok:
            print("[INFO] 使用 Raspberry Pi CSI 相機 (picamera2)")
            return cam
        cam.release()
    except Exception as e:
        print(f"[WARN] picamera2 開啟失敗：{e}")
    return None


def open_camera(preferred_index: Optional[int] = None):
    """依序嘗試開啟相機：指定索引 → Pi CSI 相機 → 手機相機 → 其餘可用相機。"""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    names = list_macos_cameras()

    candidates: list[int] = []
    if preferred_index is not None:
        candidates.append(preferred_index)
    else:
        pi_cam = open_pi_camera()
        if pi_cam is not None:
            return pi_cam
        phone_idx = find_phone_camera_index(names)
        if phone_idx is not None:
            candidates.append(phone_idx)
        if names:
            # 知道實際裝置數量時就不去試不存在的索引，免得噴一堆 OpenCV 錯誤。
            candidates += list(range(len(names)))
        else:
            # 沒有 pyobjc 可查名稱：iPhone 接上時多半排在 1，內建相機是 0。
            candidates += [1, 0]

    tried = set()
    for idx in candidates:
        if idx in tried:
            continue
        tried.add(idx)
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                label = names[idx] if idx < len(names) else "unknown"
                print(f"[INFO] 使用相機 index {idx} - {label}")
                return cap
        cap.release()
        print(f"[WARN] 相機 index {idx} 開啟失敗，嘗試下一個…")

    return None


def main():
    parser = argparse.ArgumentParser(description="物件偵測 + 追蹤 + 警報系統")
    parser.add_argument("--camera", type=int, default=None,
                        help="指定相機索引；不指定時自動優先選 Pi CSI 相機或 iPhone 接續互通相機")
    parser.add_argument("--model", default=None,
                        help="模型路徑（.hef 走 Hailo NPU、.pt 走 ultralytics）；"
                             "預設有 Hailo 就用 yolov8n.hef，否則 yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="偵測信心門檻（預設 0.4）")
    args = parser.parse_args()
    model_path = args.model or ("yolov8n.hef" if HAS_HAILO else "yolov8n.pt")

    print("=" * 50)
    print("  物件偵測 + 追蹤 + 警報系統")
    print("  滑鼠左鍵拖曳設定警戒框 | Q 結束 | R 重設追蹤")
    print("=" * 50)

    # 預設使用手機（iPhone 接續互通相機），找不到才退回其他相機。
    cap = open_camera(args.camera)
    if cap is None:
        print("[ERROR] 無法開啟相機！請確認 iPhone 已解鎖、與 Mac 同一 Apple ID，"
              "且已開啟「接續互通相機」。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        detector = Detector(model_path, conf=args.conf)
    except Exception as e:
        print(f"[ERROR] 模型載入失敗：{e}")
        cap.release()
        return

    tracker = Tracker(assoc_dist_px=160, max_miss=4)
    renderer = OverlayRenderer()
    warning_zone = WarningZone()
    alert_controller = AlertController()

    window_name = "Object Detection + Tracking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, warning_zone.mouse_callback)

    fps = 0.0
    last_time = time.time()

    print("[INFO] 相機已開啟，開始偵測...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 無法讀取畫面，重試中...")
            time.sleep(0.05)
            continue

        now_ms = time.time() * 1000

        # ---- 偵測 ----
        detections = detector.detect(frame)

        # ---- 畫偵測框（可選）----
        for (cx, cy, label, score, x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

        # ---- 追蹤 ----
        det_centers = [(cx, cy) for (cx, cy, *_) in detections]
        tracks = tracker.update(det_centers, now_ms)

        # ---- 以目前追蹤中心判斷是否已進入警戒範圍 ----
        alert = any(
            check_in_monitor(t.kf.x, t.kf.y, warning_zone.region)
            for t in tracks if t.miss == 0
        )
        alert_controller.update(alert)

        # ---- 計算 FPS ----
        now = time.time()
        dt = now - last_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst_fps
        last_time = now

        # ---- 繪製覆蓋層 ----
        renderer.draw(frame, tracks, fps, alert, warning_zone.region)

        # ---- 顯示 ----
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 使用者按下 Q，結束程式")
            break
        elif key == ord('r'):
            tracker = Tracker()
            print("[INFO] 追蹤器已重設")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
