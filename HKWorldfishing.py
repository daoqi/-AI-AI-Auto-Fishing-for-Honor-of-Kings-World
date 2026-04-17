from version import __version__
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
os.environ['ULTRALYTICS_NO_EXPLORER'] = '1'
os.environ['ULTRALYTICS_WEIGHTS_ONLY'] = '0'
import time
import random
import win32gui
import win32con
import win32api
import numpy as np
from PIL import ImageGrab
import cv2
from ultralytics import YOLO
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
    return get_resource_path(os.path.join('ai', 'best.pt'))


MODEL_PATH = get_model_path()
CAST_ROD_CONF = 0.55  # cast_rod 单独置信度
REFRESH_MS = 100
KEY_PRESS_DURATION = 0.05
CLICK_DURATION = 0.1

COOLDOWN = {
    "cast": 1.0,
    "pull": 1.0,
    "fish_bite": 0.5,
    "press_f": 0.5,
}


# ------------------------------- 辅助函数（使用整数虚拟键码）-------------------------
def click_left(x=None, y=None):
    if x is None or y is None:
        x, y = win32api.GetSystemMetrics(0) // 2, win32api.GetSystemMetrics(1) // 2
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(CLICK_DURATION)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def press_key(key):
    vk_map = {
        'w': 0x57,
        's': 0x53,
        'a': 0x41,
        'd': 0x44,
        'f': 0x46,
    }
    vk = vk_map.get(key.lower())
    if vk:
        win32api.keybd_event(vk, 0, 0, 0)
        time.sleep(KEY_PRESS_DURATION)
        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_key_multiple(key, times):
    for _ in range(times):
        press_key(key)
        time.sleep(0.05)


# ------------------------------- 获取窗口列表 -------------------------------
def get_window_list():
    windows = []

    def enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((win32gui.GetWindowText(hwnd), hwnd))

    win32gui.EnumWindows(enum_callback, None)
    return windows


# ------------------------------- 更新检查线程 -------------------------------
class UpdateChecker(QThread):
    update_available = pyqtSignal(str, str)  # latest_version, release_url

    def __init__(self, current_version):
        super().__init__()
        self.current_version = current_version

    def run(self):
        try:
            url = "https://api.github.com/repos/daoqi/-AI-AI-Auto-Fishing-for-Honor-of-Kings-World/releases/latest"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get("tag_name", "").lstrip("v")
                release_url = data.get("html_url", "")
                if latest_version and latest_version > self.current_version:
                    self.update_available.emit(latest_version, release_url)
        except Exception as e:
            print(f"检查更新失败: {e}")


# ------------------------------- 检测线程（带状态机）-------------------------
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
            self.model = YOLO(MODEL_PATH)
            self.log_message.emit("模型加载成功")
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

    def process_detections(self, detections_with_conf):
        if not self.ai_enabled:
            return

        if self.waiting_exp and (time.time() - self.last_pull_time) > self.timeout_reset:
            self.log_message.emit("超时未检测到经验，强制重置状态")
            self.waiting_exp = False
            self.exp_counted = False
            self.is_fishing = False

        try:
            # 调试：打印 cast_rod 置信度
            for label, conf in detections_with_conf:
                if label == 'cast_rod':
                    self.log_message.emit(f"cast_rod conf: {conf:.3f} (阈值: {CAST_ROD_CONF})")

            # 1. 抛竿
            if not self.is_fishing and not self.waiting_exp:
                has_fishing_stand = any(label == 'fishing_stand' for label, _ in detections_with_conf)
                has_cast_rod_high = any(
                    label == 'cast_rod' and conf >= CAST_ROD_CONF for label, conf in detections_with_conf)
                if has_fishing_stand and has_cast_rod_high:
                    if self.can_act("cast", COOLDOWN["cast"]):
                        self.log_message.emit("抛竿")
                        self.request_click_left.emit((None, None))
                        self.is_fishing = True
                        self.exp_counted = False

            # 2. 拉杆
            if self.is_fishing:
                if any(label == 'fish_on_hook' for label, _ in detections_with_conf):
                    if self.can_act("pull", COOLDOWN["pull"]):
                        self.log_message.emit("鱼上钩，拉杆")
                        self.request_click_left.emit((None, None))
                        self.is_fishing = False
                        self.waiting_exp = True
                        self.last_pull_time = time.time()

            # 3. 经验结算
            if self.waiting_exp and not self.exp_counted:
                if any(label == 'exp' for label, _ in detections_with_conf):
                    if self.can_act("exp", 0.5):
                        self.fish_count += 1
                        self.fish_count_updated.emit(self.fish_count)
                        self.log_message.emit(f"钓鱼成功！总鱼获次数: {self.fish_count}")
                        self.exp_counted = True
                        self.waiting_exp = False

            # 4. 特殊按键序列 fish_bite
            if any(label == 'fish_bite' for label, _ in detections_with_conf):
                if self.can_act("fish_bite", COOLDOWN["fish_bite"]):
                    self.log_message.emit("特殊按键序列")
                    times = random.randint(5, 8)
                    key = random.choice(['d', 'a'])
                    self.request_press_key_multiple.emit(key, times)
                    for k in ['w', 's', 'a']:
                        self.request_press_key.emit(k)
                        time.sleep(0.1)

            # 5. press_f 处理
            if any(label == 'press_f' for label, _ in detections_with_conf):
                if self.can_act("press_f", COOLDOWN["press_f"]):
                    try:
                        self.log_message.emit("按 F 键并点击中心")
                        self.request_press_key.emit('f')
                        cx, cy = win32api.GetSystemMetrics(0) // 2, win32api.GetSystemMetrics(1) // 2
                        offset_x, offset_y = random.randint(-50, 50), random.randint(-50, 50)
                        self.request_click_left.emit((cx + offset_x, cy + offset_y))
                    except Exception as e:
                        self.log_message.emit(f"press_f 执行出错: {e}")

            # 6. 快速按键序列
            rapid_actions = {
                'rapid_d': 'd',
                'rapid_a': 'a',
                'press_a': 'a',
                'press_w': 'w',
                'press_s': 's',
                'press_d': 'd'
            }
            for label, key in rapid_actions.items():
                if any(l == label for l, _ in detections_with_conf):
                    if self.can_act(f"rapid_{label}", 1.0):
                        times = random.randint(3, 5)
                        self.log_message.emit(f"快速按键 {key} {times} 次")
                        self.request_press_key_multiple.emit(key, times)
                        break

        except Exception as e:
            self.log_message.emit(f"处理检测结果异常: {e}")

    def run(self):
        if not self.init_model():
            return
        self.log_message.emit("初始化完成，等待启动检测...")
        while self.running:
            if not self.detection_enabled or self.game_rect is None:
                time.sleep(0.2)
                continue
            try:
                img = self.capture_window()
                results = self.model(img, conf=self.global_conf, verbose=False)
                detections_with_conf = []
                if results[0].boxes:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.model.names[cls]
                        detections_with_conf.append((label, conf))
                if detections_with_conf and self.ai_enabled:
                    self.log_message.emit(f"检测: {set([d[0] for d in detections_with_conf])}")
                self.process_detections(detections_with_conf)
                if self.display_enabled:
                    try:
                        annotated = results[0].plot()
                        if annotated is not None and annotated.size > 0:
                            success, encoded = cv2.imencode('.jpg', annotated)
                            if success:
                                self.frame_ready.emit(encoded.tobytes(), annotated.shape[1], annotated.shape[0])
                    except Exception as e:
                        self.log_message.emit(f"图像处理失败: {e}")
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
        conf_label = QLabel("AI识别阈值 (默认0.6):")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.6)
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
        self.update_checker = UpdateChecker(__version__)
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_checker.start()

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