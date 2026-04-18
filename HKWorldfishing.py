import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不再需要，但保留无害

from version import __version__
import sys
import time
import random
import win32gui
import win32con
import win32api
import numpy as np
from PIL import ImageGrab
import cv2
import onnxruntime as ort
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QTextEdit, QPushButton, QGroupBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
import requests
import webbrowser

# ------------------------------- 动态获取资源路径 -------------------------------
def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def get_model_path():
    return get_resource_path(os.path.join('ai', 'best.pt'))  # 改为 onnx 文件

MODEL_PATH = get_model_path()
CAST_ROD_CONF = 0.55
REFRESH_MS = 100
KEY_PRESS_DURATION = 0.05
CLICK_DURATION = 0.1

COOLDOWN = {
    "cast": 1.0,
    "pull": 1.0,
    "fish_bite": 0.5,
    "press_f": 0.5,
}

# ------------------------------- YOLO ONNX 推理类 -------------------------------
class YOLO_ONNX:
    def __init__(self, model_path, conf_thres=0.6, iou_thres=0.45):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,640,640]
        self.img_size = self.input_shape[2]  # 640

    def letterbox(self, img, new_shape=(640, 640), color=(114,114,114), auto=False, stride=32):
        # 保持宽高比缩放
        shape = img.shape[:2]  # h,w
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw // 2, dh // 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, dh
        left, right = dw, dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (dw, dh), r

    def preprocess(self, img):
        # img: BGR numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad, scale = self.letterbox(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        return img, pad, scale

    def postprocess(self, outputs, pad, scale, img_shape):
        predictions = np.transpose(outputs[0])  # (8400,84)
        boxes = []
        scores = []
        class_ids = []
        pad_x = pad[0].item() if hasattr(pad[0], 'item') else pad[0]
        pad_y = pad[1].item() if hasattr(pad[1], 'item') else pad[1]
        scale_val = scale.item() if hasattr(scale, 'item') else scale
        for pred in predictions:
            conf = np.max(pred[4:])
            if conf < self.conf_thres:
                continue
            class_id = np.argmax(pred[4:])
            x = pred[0].item()
            y = pred[1].item()
            w = pred[2].item()
            h = pred[3].item()
            # 将归一化的坐标转换为原始图片坐标
            x_center = (x - pad_x) / scale_val
            y_center = (y - pad_y) / scale_val
            width = w / scale_val
            height = h / scale_val
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(class_id)
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': scores[i],
                    'class_id': class_ids[i]
                })
        return detections
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': scores[i],
                    'class_id': class_ids[i]
                })
        return detections

    def predict(self, img):
        input_tensor, pad, scale = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, pad, scale, img.shape)
        return detections

# ------------------------------- 辅助函数（不变） -------------------------------
def click_left(x=None, y=None):
    if x is None or y is None:
        x, y = win32api.GetSystemMetrics(0)//2, win32api.GetSystemMetrics(1)//2
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(CLICK_DURATION)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def press_key(key):
    vk_map = {'w': 0x57,'s': 0x53,'a': 0x41,'d': 0x44,'f': 0x46}
    vk = vk_map.get(key.lower())
    if vk:
        win32api.keybd_event(vk, 0, 0, 0)
        time.sleep(KEY_PRESS_DURATION)
        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

def press_key_multiple(key, times):
    for _ in range(times):
        press_key(key)
        time.sleep(0.05)

def get_window_list():
    windows = []
    def enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((win32gui.GetWindowText(hwnd), hwnd))
    win32gui.EnumWindows(enum_callback, None)
    return windows

# ------------------------------- 检测线程（使用 ONNX 模型）-------------------------
class DetectionWorker(QThread):
    frame_ready = pyqtSignal(bytes, int, int)
    log_message = pyqtSignal(str)
    fish_count_updated = pyqtSignal(int)
    request_click_left = pyqtSignal(tuple)
    request_press_key = pyqtSignal(str)
    request_press_key_multiple = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.detection_enabled = False
        self.ai_enabled = False
        self.display_enabled = True
        self.model = None
        self.game_hwnd = None
        self.game_rect = None
        self.win_w = self.win_h = 0
        self.last_action = {}
        self.fish_count = 0
        self.struggling = False  # 是否正在挣扎
        self.struggle_start_time = 0  # 挣扎开始时间
        self._last_struggle_key = 0  # 上次挣扎按键时间

        self.is_fishing = False
        self.waiting_exp = False
        self.exp_counted = False
        self.last_pull_time = 0

        self.global_conf = 0.6
        self.timeout_reset = 45

    def set_game_window(self, hwnd):
        if not hwnd:
            return False
        self.game_hwnd = hwnd
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        rect = win32gui.GetWindowRect(hwnd)
        x, y, right, bottom = rect
        self.win_w, self.win_h = right - x, bottom - y
        self.game_rect = (x, y, right, bottom)
        self.log_message.emit(f"选中窗口: {win32gui.GetWindowText(hwnd)} 大小: {self.win_w}x{self.win_h}")
        if self.win_w < 200 or self.win_h < 200:
            self.log_message.emit("窗口尺寸异常，请确保窗口正常显示")
            return False
        return True

    def init_model(self):
        try:
            self.model = YOLO_ONNX(MODEL_PATH, conf_thres=self.global_conf, iou_thres=0.45)
            self.log_message.emit("ONNX 模型加载成功")
            return True
        except Exception as e:
            self.log_message.emit(f"模型加载失败: {e}")
            return False

    def capture_window(self):
        x, y, right, bottom = self.game_rect
        img = ImageGrab.grab(bbox=(x, y, right, bottom))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def can_act(self, action, cooldown):
        now = time.time()
        last = self.last_action.get(action, 0)
        if now - last >= cooldown:
            self.last_action[action] = now
            return True
        return False

    def process_detections(self, detections):
        if not self.ai_enabled:
            return
        if self.waiting_exp and (time.time() - self.last_pull_time) > self.timeout_reset:
            self.log_message.emit("超时未检测到经验，强制重置状态")
            self.waiting_exp = False
            self.exp_counted = False
            self.is_fishing = False
        try:
            # 获取检测结果中的标签和置信度
            labels_confs = [(self.model.names[d['class_id']], d['confidence']) for d in detections]
            if labels_confs and self.ai_enabled:
                self.log_message.emit(f"检测: {set([l for l,_ in labels_confs])}")
            # 1. 抛竿：检测到 fishing_stand 和 cast_rod 就抛竿（无条件）
            if not self.struggling:
                has_fishing_stand = any(l == 'fishing_stand' for l, _ in labels_confs)
                has_cast_rod_high = any(l == 'cast_rod' and conf >= CAST_ROD_CONF for l, conf in labels_confs)
                if has_fishing_stand and has_cast_rod_high:
                    if self.can_act("cast", COOLDOWN["cast"]):
                        self.log_message.emit("抛竿")
                        self.request_click_left.emit((None, None))
                        self.is_fishing = True
                        self.waiting_exp = False
                        self.exp_counted = False
                    else:
                        self.log_message.emit("[DEBUG] 抛竿冷却中")
                else:
                    # 可选：打印未满足条件的原因
                    if not has_fishing_stand:
                        self.log_message.emit("[DEBUG] 未检测到 fishing_stand")
                    elif not has_cast_rod_high:
                        cast_rod_conf = next((conf for l, conf in labels_confs if l == 'cast_rod'), None)
                        if cast_rod_conf is not None:
                            self.log_message.emit(
                                f"[DEBUG] cast_rod 置信度 {cast_rod_conf:.3f} 低于阈值 {CAST_ROD_CONF}")
                        else:
                            self.log_message.emit("[DEBUG] 未检测到 cast_rod")

            # 2. 拉杆
            if self.is_fishing:
                if any(l == 'fish_on_hook' for l, _ in labels_confs):
                    if self.can_act("pull", COOLDOWN["pull"]):
                        self.log_message.emit("鱼上钩，拉杆")
                        self.request_click_left.emit((None, None))
                        self.is_fishing = False
                        self.waiting_exp = True
                        self.last_pull_time = time.time()
                        self.struggling = False  # 拉杆成功，结束挣扎

            # 3. 经验结算
            if self.waiting_exp and not self.exp_counted:
                if any(l == 'exp' for l, _ in labels_confs):
                    if self.can_act("exp", 0.5):
                        self.fish_count += 1
                        self.fish_count_updated.emit(self.fish_count)
                        self.log_message.emit(f"钓鱼成功！总鱼获次数: {self.fish_count}")
                        self.exp_counted = True
                        self.waiting_exp = False
                        self.struggling = False

            # 4. fish_bite
            if any(l == 'fish_bite' for l, _ in labels_confs):
                if not self.struggling and self.can_act("fish_bite", COOLDOWN["fish_bite"]):
                    self.log_message.emit("钓大鱼需要操作，开始挣扎处理")
                    self.struggling = True
                    self.struggle_start_time = time.time()
                    # 立即执行一次初始按键
                    key = random.choice(['d', 'a'])
                    self.request_press_key_multiple.emit(key, random.randint(3, 5))

            # 5. press_f
            if any(l == 'press_f' for l,_ in labels_confs):
                if self.can_act("press_f", COOLDOWN["press_f"]):
                    try:
                        self.log_message.emit("按 F 键并点击中心")
                        self.request_press_key.emit('f')
                        cx, cy = win32api.GetSystemMetrics(0)//2, win32api.GetSystemMetrics(1)//2
                        offset_x, offset_y = random.randint(-50, 50), random.randint(-50, 50)
                        self.request_click_left.emit((cx+offset_x, cy+offset_y))
                    except Exception as e:
                        self.log_message.emit(f"press_f 执行出错: {e}")

            rapid_actions = {
                'rapid_d': 'd', 'rapid_a': 'a', 'press_a': 'a',
                'press_w': 'w', 'press_s': 's', 'press_d': 'd'
            }
            for label, key in rapid_actions.items():
                if any(l == label for l, _ in labels_confs):
                    if self.can_act(f"rapid_{label}", 1.0):
                        times = random.randint(3, 5)
                        self.log_message.emit(f"快速按键 {key} {times} 次")
                        self.request_press_key_multiple.emit(key, times)
                        break

            # ========= 挣扎状态持续处理 =========
            if self.struggling:
                elapsed = time.time() - self.struggle_start_time
                fish_on_hook_exists = any(l == 'fish_on_hook' for l, _ in labels_confs)
                exp_exists = any(l == 'exp' for l, _ in labels_confs)
                stand_or_rod = any(l in ('fishing_stand', 'cast_rod') for l, _ in labels_confs)

                # 结束挣扎的条件：成功经验、脱钩（出现钓鱼台或抛竿）、鱼上钩消失、或保底超时15秒
                if exp_exists or stand_or_rod or not fish_on_hook_exists or elapsed > 15.0:
                    if elapsed > 15.0:
                        self.log_message.emit("挣扎保底超时（15秒），强制结束")
                    else:
                        self.log_message.emit("挣扎结束（成功或脱钩）")
                    self.struggling = False
                    if stand_or_rod or not fish_on_hook_exists:
                        self.is_fishing = False
                        self.waiting_exp = False
                else:
                    # 持续快速按键，每 0.08 秒执行一次
                    if time.time() - self._last_struggle_key > 0.08:
                        self._last_struggle_key = time.time()
                        # 优先根据检测到的挣扎方向按键
                        if any(l == 'rapid_d' for l, _ in labels_confs):
                            key = 'd'
                        elif any(l == 'rapid_a' for l, _ in labels_confs):
                            key = 'a'
                        else:
                            key = random.choice(['d', 'a'])
                        times = random.randint(5, 8)
                        self.log_message.emit(f"挣扎中：快速按键 {key} {times} 次")
                        self.request_press_key_multiple.emit(key, times)
            # ================================================================

        except Exception as e:
            self.log_message.emit(f"处理检测结果异常: {e}")

    def run(self):
        if not self.init_model():
            return
        self.log_message.emit("初始化完成，等待启动检测...")
        # 加载类别名称（从 ONNX 模型元数据获取，若无则手动定义）
        self.model.names = [
            'fishing_stand', 'cast_rod', 'fish_bite', 'rapid_d', 'rapid_a',
            'press_f', 'exp', 'fish_on_hook', 'pull_rod', 'press_w',
            'press_s', 'press_a', 'press_d'
        ]
        while self.running:
            if not self.detection_enabled or self.game_rect is None:
                time.sleep(0.2)
                continue
            try:
                img = self.capture_window()
                detections = self.model.predict(img)
                # 可选：绘制检测框用于显示
                if self.display_enabled:
                    annotated = img.copy()
                    for d in detections:
                        x1,y1,x2,y2 = d['bbox']
                        label = self.model.names[d['class_id']]
                        conf = d['confidence']
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(annotated, f"{label} {conf:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    success, encoded = cv2.imencode('.jpg', annotated)
                    if success:
                        self.frame_ready.emit(encoded.tobytes(), annotated.shape[1], annotated.shape[0])
                self.process_detections(detections)
            except Exception as e:
                self.log_message.emit(f"检测循环异常: {e}")
            time.sleep(REFRESH_MS / 1000.0)

    def stop(self):
        self.running = False
        self.wait()

    def set_detection_enabled(self, enabled):
        self.detection_enabled = enabled
        self.log_message.emit(f"检测脚本{'已启动' if enabled else '已停止'}")
        if not enabled:
            self.is_fishing = False
            self.waiting_exp = False
            self.exp_counted = False
            self.last_pull_time = 0

    def set_ai_enabled(self, enabled):
        self.ai_enabled = enabled
        self.log_message.emit(f"AI 自动操作{'已启用' if enabled else '已禁用'}")
        if not enabled:
            self.is_fishing = False
            self.waiting_exp = False
            self.exp_counted = False
            self.last_pull_time = 0

    def set_display_enabled(self, enabled):
        self.display_enabled = enabled
        self.log_message.emit(f"画面显示{'已开启' if enabled else '已关闭'}")

# ------------------------------- 主UI（基本不变）-------------------------

# ------------------------------- 主UI --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"王者荣耀世界AI钓鱼  v{__version__}")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0f1a; }
            QGroupBox {
                font: bold 14px; color: #0ff;
                border: 1px solid #0ff; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
                background-color: rgba(0, 255, 255, 0.05);
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QLabel { color: #0ff; background-color: rgba(0,0,0,0.5); border: 1px solid #0ff; border-radius: 4px; padding: 2px; }
            QCheckBox { color: #0ff; font-size: 14px; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 3px; border: 1px solid #0ff; background-color: #0a0f1a; }
            QCheckBox::indicator:checked { background-color: #0ff; }
            QTextEdit { background-color: #0a0f1a; color: #0ff; border: 1px solid #0ff; border-radius: 4px; font-family: Consolas; }
            QPushButton { background-color: #0a0f1a; color: #0ff; border: 1px solid #0ff; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #0ff; color: #0a0f1a; }
            QComboBox { background-color: #0a0f1a; color: #0ff; border: 1px solid #0ff; border-radius: 4px; padding: 2px; }
        """)

        icon_path = get_resource_path('logo.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.worker = DetectionWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_message.connect(self.append_log)
        self.worker.fish_count_updated.connect(self.update_fish_count)

        self.worker.request_click_left.connect(self.do_click_left)
        self.worker.request_press_key.connect(self.do_press_key)
        self.worker.request_press_key_multiple.connect(self.do_press_key_multiple)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid #0ff; background-color: #000;")
        layout.addWidget(self.video_label, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        win_group = QGroupBox("窗口选择")
        win_layout = QVBoxLayout()
        self.window_combo = QComboBox()
        self.refresh_btn = QPushButton("刷新窗口列表")
        self.refresh_btn.clicked.connect(self.refresh_window_list)
        win_layout.addWidget(self.window_combo)
        win_layout.addWidget(self.refresh_btn)
        win_group.setLayout(win_layout)
        right_layout.addWidget(win_group)

        ctrl_group = QGroupBox("脚本控制 (F12全局)")
        ctrl_layout = QVBoxLayout()
        self.start_btn = QPushButton("启动检测")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_group.setLayout(ctrl_layout)
        right_layout.addWidget(ctrl_group)

        ai_group = QGroupBox("AI 操作 (F11全局)")
        ai_layout = QVBoxLayout()
        self.ai_checkbox = QCheckBox("启用 AI  ")
        self.ai_checkbox.setChecked(False)
        self.ai_checkbox.stateChanged.connect(self.on_ai_toggled)
        ai_layout.addWidget(self.ai_checkbox)
        ai_group.setLayout(ai_layout)
        right_layout.addWidget(ai_group)

        display_group = QGroupBox("画面显示")
        display_layout = QVBoxLayout()
        self.display_checkbox = QCheckBox("显示AI视角")
        self.display_checkbox.setChecked(True)
        self.display_checkbox.stateChanged.connect(self.on_display_toggled)
        display_layout.addWidget(self.display_checkbox)
        display_group.setLayout(display_layout)
        right_layout.addWidget(display_group)

        top_group = QGroupBox("窗口设置")
        top_layout = QVBoxLayout()
        self.game_top_checkbox = QCheckBox("游戏窗口置顶")
        self.game_top_checkbox.setChecked(False)
        self.game_top_checkbox.stateChanged.connect(self.on_game_top_toggled)
        top_layout.addWidget(self.game_top_checkbox)
        top_group.setLayout(top_layout)
        right_layout.addWidget(top_group)

        settings_group = QGroupBox("AI 参数设置")
        settings_layout = QVBoxLayout()

        conf_layout = QHBoxLayout()
        conf_label = QLabel("AI识别阈值 (默认0.5):")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_spin)
        settings_layout.addLayout(conf_layout)

        timeout_layout = QHBoxLayout()
        timeout_label = QLabel("AI超时重置 (默认45):")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 300)
        self.timeout_spin.setValue(45)
        self.timeout_spin.valueChanged.connect(self.on_timeout_changed)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_spin)
        settings_layout.addLayout(timeout_layout)

        settings_group.setLayout(settings_layout)
        right_layout.addWidget(settings_group)

        stats_group = QGroupBox("钓鱼统计")
        stats_layout = QVBoxLayout()
        self.fish_label = QLabel("当前鱼获次数: 0")
        self.fish_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.fish_label)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        log_group = QGroupBox("事件日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(clear_btn)
        right_layout.addLayout(btn_layout)

        right_layout.addStretch()
        layout.addWidget(right, 1)

        self.hotkey_f12 = 1
        self.hotkey_f11 = 2
        try:
            win32gui.RegisterHotKey(int(self.winId()), self.hotkey_f12, 0, win32con.VK_F12)
            win32gui.RegisterHotKey(int(self.winId()), self.hotkey_f11, 0, win32con.VK_F11)
            self.append_log("全局热键已注册: F12 启动/停止脚本, F11 开关 AI 操作")
        except Exception as e:
            self.append_log(f"注册热键失败: {e}，请以管理员身份运行程序")

        self.ai_enable_timer = None
        self.ai_countdown = 0

        self.worker.start()
        self.refresh_window_list()
        self.append_log("AI已启动，请选择游戏窗口，然后点击「启动检测」或按 F12。")
        self.append_log("获取最新AI版本加入QQ群1098948146")
        self.append_log("AI阈值不要超过0.9 AI重置不能低于45")

        # 启动更新检查（非阻塞）
        # self.update_checker = UpdateChecker(__version__)
        # self.update_checker.update_available.connect(self.on_update_available)
        # self.update_checker.start()

    # 主线程执行的底层操作
    def do_click_left(self, pos):
        x, y = pos
        if x is None:
            click_left()
        else:
            click_left(x, y)

    def do_press_key(self, key):
        press_key(key)

    def do_press_key_multiple(self, key, times):
        press_key_multiple(key, times)

    def nativeEvent(self, eventType, message):
        try:
            msg = message
            if msg.message == win32con.WM_HOTKEY:
                hotkey_id = msg.wParam
                if hotkey_id == self.hotkey_f12:
                    self.toggle_detection()
                elif hotkey_id == self.hotkey_f11:
                    self.toggle_ai()
                return True, 0
        except:
            pass
        return super().nativeEvent(eventType, message)

    def refresh_window_list(self):
        self.window_combo.clear()
        windows = get_window_list()
        for title, hwnd in windows:
            self.window_combo.addItem(f"{title} (0x{hwnd:X})", hwnd)
        if self.window_combo.count() > 0:
            self.append_log(f"已找到 {self.window_combo.count()} 个窗口，请选择游戏窗口。")

    def start_detection(self):
        idx = self.window_combo.currentIndex()
        if idx < 0:
            self.append_log("请先选择一个窗口！")
            return
        hwnd = self.window_combo.currentData()
        if not self.worker.set_game_window(hwnd):
            self.append_log("窗口设置失败，请重新选择")
            return
        self.worker.set_detection_enabled(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.window_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        if self.game_top_checkbox.isChecked():
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    def stop_detection(self):
        self.worker.set_detection_enabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.window_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)

    def toggle_detection(self):
        if self.start_btn.isEnabled() and not self.stop_btn.isEnabled():
            self.start_detection()
        else:
            self.stop_detection()

    def toggle_ai(self):
        new_state = not self.ai_checkbox.isChecked()
        self.ai_checkbox.setChecked(new_state)

    def on_ai_toggled(self, state):
        if state == Qt.Checked:
            self.start_ai_delay()
        else:
            self.cancel_ai_delay()
            self.worker.set_ai_enabled(False)

    def start_ai_delay(self):
        self.cancel_ai_delay()
        self.append_log("AI自动操作将在3秒后启用，请切至游戏窗口...")
        self.ai_countdown = 3
        self.ai_enable_timer = QTimer()
        self.ai_enable_timer.timeout.connect(self.ai_countdown_tick)
        self.ai_enable_timer.start(1000)

    def ai_countdown_tick(self):
        self.ai_countdown -= 1
        if self.ai_countdown > 0:
            self.append_log(f"{self.ai_countdown}")
        else:
            self.ai_enable_timer.stop()
            self.ai_enable_timer = None
            self.worker.set_ai_enabled(True)
            self.append_log("AI自动操作已启用")

    def cancel_ai_delay(self):
        if self.ai_enable_timer is not None:
            self.ai_enable_timer.stop()
            self.ai_enable_timer = None

    def on_display_toggled(self, state):
        enabled = (state == Qt.Checked)
        self.worker.set_display_enabled(enabled)
        self.append_log(f"画面显示已{'开启' if enabled else '关闭'}，AI自动操作{'仍会继续' if not enabled else '正常'}")

    def on_game_top_toggled(self, state):
        enabled = (state == Qt.Checked)
        hwnd = self.window_combo.currentData()
        if not hwnd:
            self.append_log("请先选择游戏窗口！")
            self.game_top_checkbox.setChecked(False)
            return
        if enabled:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            self.append_log("游戏窗口已置顶")
        else:
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            self.append_log("游戏窗口已取消置顶")

    def on_conf_changed(self, value):
        self.worker.global_conf = value
        self.append_log(f"AI识别阈值已更改为: {value:.2f}")

    def on_timeout_changed(self, value):
        self.worker.timeout_reset = value
        self.append_log(f"AI超时重置时间已更改为: {value} 秒")

    def on_update_available(self, latest_version, release_url):
        reply = QMessageBox.question(
            self,
            "发现新版本",
            f"当前版本 v{__version__}\n最新版本 v{latest_version}\n\n是否前往下载更新？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            webbrowser.open(release_url)

    def update_frame(self, img_bytes, w, h):
        qimg = QImage.fromData(img_bytes)
        if qimg.isNull():
            return
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def append_log(self, msg):
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        scroll = self.log_text.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def clear_log(self):
        self.log_text.clear()

    def update_fish_count(self, count):
        self.fish_label.setText(f"当前鱼获次数: {count}")

    def closeEvent(self, event):
        try:
            win32gui.UnregisterHotKey(int(self.winId()), self.hotkey_f12)
            win32gui.UnregisterHotKey(int(self.winId()), self.hotkey_f11)
        except:
            pass
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())