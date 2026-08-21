"""Modern PyQt5 dashboard for the "Juancha Guard" fatigue detection project.

The window can run without OpenCV/PyTorch so the UI can be reviewed separately.
Detection code can push frames and values through update_frame/update_metrics.
"""

import os
import sys
import traceback
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)

from core_engine import (
    ContinuousFatigueDetectionThread,
    FaceRecognitionThread,
    FatigueCheckInThread,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
# 科技感安全监控主题色
BLUE = "#1890ff"
CYAN = "#13c2c2"
GREEN = "#52c41a"
RED = "#ff4d4f"
DARK_BG = "#0d1117"
CARD_BG = "#161b22"

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        # 优化阴影效果，使UI更具现代悬浮感
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)


class NavButton(QPushButton):
    def __init__(self, icon, text):
        super().__init__(f"{icon}   {text}")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("navButton")


class MetricCard(Card):
    def __init__(self, icon, title, value, suffix, accent, detail):
        super().__init__()
        self.setMinimumHeight(126)
        row = QHBoxLayout(self)
        row.setContentsMargins(22, 18, 22, 18)

        badge = QLabel(icon)
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(54, 54)
        badge.setStyleSheet(
            f"background:{accent}22;color:{accent};border:1px solid {accent}55;"
            "border-radius:12px;font-size:24px;font-weight:bold;"
        )
        row.addWidget(badge)

        texts = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("muted")

        number = QHBoxLayout()
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size:28px;font-weight:800;color:#c9d1d9;")
        unit = QLabel(suffix)
        unit.setObjectName("muted")
        number.addWidget(self.value_label)
        number.addWidget(unit, 0, Qt.AlignBottom)
        number.addStretch()

        detail_label = QLabel(detail)
        detail_label.setStyleSheet(f"color:{accent};font-size:12px;")

        texts.addWidget(title_label)
        texts.addLayout(number)
        texts.addWidget(detail_label)
        row.addLayout(texts, 1)


class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.setObjectName("cameraView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(650, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source = None
        self.setText("CAMERA 01\n\n等待 USB 摄像头信号接入...")

    def set_frame(self, frame):
        if isinstance(frame, QImage):
            image = frame
        else:
            h, w = frame.shape[:2]
            image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888).copy()
        self._source = QPixmap.fromImage(image)
        self._refresh()

    def load_demo_image(self):
        for name in ("test_done.jpg", "test.jpg", "result.jpg"):
            path = os.path.join(ROOT, name)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._source = pixmap
                self._refresh()
                return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._source:
            self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class PlaceholderPage(QWidget):
    def __init__(self, title, subtitle):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(42, 34, 42, 34)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        copy = QLabel(subtitle)
        copy.setObjectName("muted")
        layout.addWidget(heading)
        layout.addWidget(copy)
        layout.addStretch()


class DashboardPage(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 26, 32, 28)
        root.setSpacing(18)

        # Header 区域
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("精细作业环境打卡系统")
        title.setObjectName("pageTitle")
        subtitle = QLabel("并行身份认证与实时生理状态监测 (EAR/MAR)")
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.clock = QLabel()
        self.clock.setObjectName("clock")
        online = QLabel("●  算法引擎在线")
        online.setStyleSheet(f"color:{GREEN};font-size:13px;font-weight:bold;")
        header.addWidget(self.clock)
        header.addSpacing(18)
        header.addWidget(online)
        root.addLayout(header)

        # 主体布局 (左侧视频 + 右侧控制与数据)
        body = QHBoxLayout()
        body.setSpacing(18)

        # 视频卡片
        video_card = Card()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)
        video_bar = QHBoxLayout()
        live = QLabel("●  LIVE")
        live.setStyleSheet(f"color:{RED};font-weight:900;letter-spacing:1px;")
        camera_name = QLabel("打卡终端 USB 摄像头 / 1920 × 1080")
        camera_name.setObjectName("muted")
        video_bar.addWidget(live)
        video_bar.addStretch()
        video_bar.addWidget(camera_name)
        self.camera = CameraView()
        video_layout.addLayout(video_bar)
        video_layout.addWidget(self.camera, 1)
        body.addWidget(video_card, 7)

        # 右侧面板
        side = QVBoxLayout()
        side.setSpacing(14)

        # 认证与状态面板
        status = Card()
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(22, 22, 22, 22)
        status_layout.setSpacing(13)
        status_title = QLabel("实时认证判定")
        status_title.setObjectName("sectionTitle")

        self.status_badge = QLabel("✓")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedSize(58, 58)
        self.status_badge.setStyleSheet(
            f"border:2px solid {GREEN};border-radius:29px;color:{GREEN};font-size:30px;"
        )
        self.status_text = QLabel("允许打卡/上岗")
        self.status_text.setStyleSheet(f"color:{GREEN};font-size:24px;font-weight:700;")
        state_row = QHBoxLayout()
        state_row.addWidget(self.status_badge)
        state_row.addSpacing(8)
        state_row.addWidget(self.status_text)
        state_row.addStretch()

        status_layout.addWidget(status_title)
        status_layout.addLayout(state_row)
        # 对齐论文中的人脸特征比对与状态判定
        self.current_user = self._data_row("当前认证人员", "等待检测")
        status_layout.addWidget(self.current_user)
        self.confidence = self._data_row("人脸匹配相似度", "--")
        status_layout.addWidget(self.confidence)
        self.fps = self._data_row("处理帧率", "-- FPS")
        status_layout.addWidget(self.fps)
        side.addWidget(status)

        # 特征参数面板
        feature = Card()
        feature_layout = QVBoxLayout(feature)
        feature_layout.setContentsMargins(22, 20, 22, 20)
        feature_layout.setSpacing(13)
        heading = QLabel("生理特征监控 (双阈值)")
        heading.setObjectName("sectionTitle")
        feature_layout.addWidget(heading)
        self.eye_value = QLabel("EAR: 0.32 (正常)")
        self.mouth_value = QLabel("MAR: 0.21 (闭合)")
        self.perclos_value = QLabel("0.00")
        for label, widget in (
            ("眼部状态", self.eye_value),
            ("嘴部状态", self.mouth_value),
            ("PERCLOS", self.perclos_value),
        ):
            feature_layout.addWidget(self._feature_row(label, widget))
        side.addWidget(feature)

        # 控制按钮
        controls = Card()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        self.start_btn = QPushButton("▶  开始识别打卡")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.setObjectName("secondaryButton")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        controls_layout.addWidget(self.start_btn, 2)
        controls_layout.addWidget(self.stop_btn, 1)
        side.addWidget(controls)
        side.addStretch()

        body.addLayout(side, 3)
        root.addLayout(body, 1)

        # 底部指标卡片
        metrics = QHBoxLayout()
        metrics.setSpacing(14)
        self.duration = MetricCard("◷", "5秒判定窗口", "00:00", "", BLUE, "当前打卡耗时")
        self.blinks = MetricCard("◉", "连续闭眼帧数", "0", "帧", CYAN, "阈值: ≥15帧(约0.6s)")
        self.yawns = MetricCard("◇", "频繁眨眼/哈欠", "0", "次", "#b58cff", "加权疲劳特征统计")
        self.risk = MetricCard("!", "综合评估结果", "通过", "", GREEN, "结合身份与生理状态")
        for card in (self.duration, self.blinks, self.yawns, self.risk):
            metrics.addWidget(card)
        root.addLayout(metrics)

        self.elapsed = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._clock)
        self.clock_timer.start(1000)
        self._clock()
        self.camera.load_demo_image()

    def _data_row(self, key, value):
        box = QFrame()
        box.setObjectName("dataRow")
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 9, 12, 9)
        left = QLabel(key)
        left.setObjectName("muted")
        right = QLabel(value)
        right.setObjectName("dataValue")
        row.addWidget(left)
        row.addStretch()
        row.addWidget(right)
        box.value_label = right
        return box

    def _feature_row(self, title, value):
        row = QFrame()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 5, 0, 5)
        label = QLabel(title)
        label.setObjectName("muted")
        value.setStyleSheet(f"color:{GREEN};font-weight:700;")
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        return row

    def _clock(self):
        self.clock.setText(datetime.now().strftime("%Y-%m-%d   %H:%M:%S"))

    def _start(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(1000)
        self.start_requested.emit()

    def _stop(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.timer.stop()
        self.stop_requested.emit()

    def _tick(self):
        self.elapsed += 1
        minutes, seconds = divmod(self.elapsed, 60)
        self.duration.value_label.setText(f"{minutes:02d}:{seconds:02d}")

    def update_frame(self, frame):
        self.camera.set_frame(frame)

    def update_metrics(self, *, blinks=None, yawns=None, current_user=None, fps=None,
                       confidence=None, eye_state=None, mouth_state=None, risk=None):
        if blinks is not None:
            self.blinks.value_label.setText(str(blinks))
        if yawns is not None:
            self.yawns.value_label.setText(str(yawns))
        if current_user is not None:
            self.current_user.value_label.setText(current_user)
        if fps is not None:
            self.fps.value_label.setText(f"{fps:.1f} FPS")
        if confidence is not None:
            self.confidence.value_label.setText(f"{confidence:.1%}")
        if eye_state is not None:
            self.eye_value.setText(eye_state)
        if mouth_state is not None:
            self.mouth_value.setText(mouth_state)
        if risk is not None:
            high_risk = risk in ("疲劳", "未注册", "拒绝", True)
            color = RED if high_risk else GREEN
            self.risk.value_label.setText("拦截" if high_risk else "通过")
            self.status_text.setText("状态异常/拒绝" if high_risk else "允许打卡/上岗")
            self.status_badge.setText("×" if high_risk else "✓")
            self.status_badge.setStyleSheet(
                f"border:2px solid {color};border-radius:29px;color:{color};font-size:30px;"
            )
            self.status_text.setStyleSheet(f"color:{color};font-size:24px;font-weight:700;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.current_mode = "checkin"
        self.setWindowTitle("倦察卫士 - 精细作业环境打卡系统")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 760)
        shell = QWidget()
        self.setCentralWidget(shell)
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧侧边栏导航
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(16, 24, 16, 22)
        nav.setSpacing(8)

        brand = QLabel("◈  倦察卫士")
        brand.setObjectName("brand")
        tagline = QLabel("JUANCHA GUARD SYSTEM")
        tagline.setObjectName("brandSub")
        nav.addWidget(brand)
        nav.addWidget(tagline)
        nav.addSpacing(28)

        self.buttons = []
        # 新增“告警事件”一栏，与打卡记录分离开
        entries = [
            ("▣", "疲劳检测打卡模式"),
            ("◎", "持续疲劳检测模式"),
            ("▤", "纯人脸认证模式"),
            ("▤", "打卡记录"),
            ("△", "告警事件"),
            ("⚙", "系统阈值设置")
        ]
        for index, (icon, text) in enumerate(entries):
            button = NavButton(icon, text)
            button.clicked.connect(lambda checked, i=index: self.select_page(i))
            nav.addWidget(button)
            self.buttons.append(button)

        nav.addStretch()
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")
        nav.addWidget(divider)
        user = QLabel("●   本地终端节点\n     管理员权限")
        user.setObjectName("user")
        nav.addWidget(user)

        self.pages = QStackedWidget()
        self.dashboard = DashboardPage()
        self.pages.addWidget(self.dashboard)

        # 为新添加的侧边栏项预留页面占位符
        for title in ("持续疲劳检测模式", "纯人脸认证模式", "打卡记录", "告警事件", "系统阈值设置"):
            self.pages.addWidget(
                PlaceholderPage(title, f"已进入 {title} 页面。底层模型将根据 QStackedWidget 视图切换工作流。"))

        layout.addWidget(sidebar)
        layout.addWidget(self.pages, 1)
        self.dashboard.start_requested.connect(self.start_detection)
        self.dashboard.stop_requested.connect(self.stop_detection)
        self.select_page(0)

    def select_page(self, index):
        if index < 3:
            requested_mode = ("checkin", "continuous", "recognition")[index]
            if requested_mode != self.current_mode:
                self.stop_detection()
            self.current_mode = requested_mode
            self.pages.setCurrentIndex(0)
        else:
            self.stop_detection()
            self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)

    def start_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            return
        thread_types = {
            "checkin": FatigueCheckInThread,
            "continuous": ContinuousFatigueDetectionThread,
            "recognition": FaceRecognitionThread,
        }
        try:
            self.detection_thread = thread_types[self.current_mode]()
            self.detection_thread.change_pixmap_signal.connect(self.dashboard.update_frame)
            self.detection_thread.status_update_signal.connect(self.handle_status_update)
            self.detection_thread.finished.connect(self.thread_finished)
            self.detection_thread.start()
        except Exception as exc:
            self.detection_thread = None
            self.dashboard._stop()
            QMessageBox.critical(self, "启动失败", str(exc))

    def stop_detection(self):
        thread = self.detection_thread
        self.detection_thread = None
        if thread and thread.isRunning():
            thread.stop()
        if self.dashboard.timer.isActive():
            self.dashboard.timer.stop()
        self.dashboard.start_btn.setEnabled(True)
        self.dashboard.stop_btn.setEnabled(False)

    def thread_finished(self):
        self.detection_thread = None
        self.dashboard.start_btn.setEnabled(True)
        self.dashboard.stop_btn.setEnabled(False)
        self.dashboard.timer.stop()

    def handle_status_update(self, status):
        if status.get("error"):
            self.dashboard.update_metrics(risk=True)
            self.dashboard.status_text.setText(status["error"])
            return

        metrics = {"fps": status.get("fps")}
        if self.current_mode == "continuous":
            metrics.update(
                blinks=status.get("blink_count"),
                yawns=status.get("yawn_count"),
                eye_state=status.get("eye_state"),
                mouth_state=status.get("mouth_state"),
                risk=status.get("fatigue", False),
            )
            if status.get("perclos") is not None:
                self.dashboard.perclos_value.setText(f"{status['perclos']:.2f}")
        elif self.current_mode == "recognition":
            identity = status.get("last_recognized_identity")
            metrics.update(
                current_user=identity or "未识别",
                confidence=status.get("confidence"),
                risk="未注册" if identity in (None, "Unknown", "未检测到人脸") else False,
            )
        else:
            result = status.get("evaluation_result") or {}
            identity = status.get("identity", "未检测到")
            metrics.update(
                blinks=status.get("blink_count"),
                yawns=status.get("yawn_count"),
                current_user=identity,
                confidence=status.get("confidence"),
                eye_state=status.get("eye_state"),
                mouth_state=status.get("mouth_state"),
                risk=result.get("fatigue", False) if result else identity in ("未检测到", "Unknown"),
            )
            perclos = result.get("perclos", status.get("perclos"))
            if perclos is not None:
                self.dashboard.perclos_value.setText(f"{perclos:.2f}")
        self.dashboard.update_metrics(**{key: value for key, value in metrics.items() if value is not None})

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


# 深度优化的 QSS：增加平滑、现代扁平的视觉体验，优化控件层次感
STYLE = f"""
* {{ font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; font-size: 14px; color: #c9d1d9; }}
QMainWindow, QWidget {{ background: {DARK_BG}; }}
QFrame#sidebar {{ background: {CARD_BG}; border-right: 1px solid #30363d; }}

QLabel#brand {{ font-size: 24px; font-weight: 800; color: #f0f6fc; padding: 2px 8px; letter-spacing: 1px; }}
QLabel#brandSub {{ font-size: 10px; color: #8b949e; padding-left: 40px; letter-spacing: 0.5px; font-weight: bold; }}

QPushButton#navButton {{ border: 0; border-radius: 8px; background: transparent; text-align: left; padding: 15px 18px; color: #8b949e; font-size: 15px; font-weight: 600; outline: none; }}
QPushButton#navButton:hover {{ background: #21262d; color: #c9d1d9; }}
QPushButton#navButton:checked {{ background: #1f2937; color: {BLUE}; border-left: 4px solid {BLUE}; border-top-left-radius: 0px; border-bottom-left-radius: 0px; }}

QFrame#divider {{ color: #30363d; }}
QLabel#user {{ color: #8b949e; padding: 12px 8px; line-height: 1.6; font-size: 13px; }}

QLabel#pageTitle {{ font-size: 26px; font-weight: 800; color: #f0f6fc; letter-spacing: 1px; }}
QLabel#sectionTitle {{ font-size: 17px; font-weight: 700; color: #c9d1d9; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
QLabel#muted {{ color: #8b949e; font-size: 13px; }}
QLabel#clock {{ color: #8b949e; font-family: 'Consolas', monospace; font-size: 14px; background: #21262d; padding: 4px 10px; border-radius: 6px; }}

QFrame#card {{ background: {CARD_BG}; border: 1px solid #30363d; border-radius: 12px; }}
QLabel#cameraView {{ background: #010409; border: 1px solid #30363d; border-radius: 8px; color: #484f58; font-size: 16px; font-weight: bold; }}

QFrame#dataRow {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; }}
QLabel#dataValue {{ color: #f0f6fc; font-weight: 700; font-size: 15px; }}

QPushButton {{ min-height: 44px; border-radius: 8px; font-weight: 700; font-size: 15px; padding: 0 20px; outline: none; }}
QPushButton#primaryButton {{ background: {BLUE}; color: #ffffff; border: 1px solid #1890ff; }}
QPushButton#primaryButton:hover {{ background: #40a9ff; border-color: #40a9ff; }}
QPushButton#primaryButton:pressed {{ background: #096dd9; border-color: #096dd9; }}

QPushButton#secondaryButton {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }}
QPushButton#secondaryButton:hover {{ background: #30363d; border-color: #8b949e; }}
QPushButton#secondaryButton:pressed {{ background: #161b22; }}
QPushButton:disabled {{ background: #161b22; color: #484f58; border-color: #30363d; }}
"""


def main():
    def report_exception(exc_type, exc_value, exc_traceback):
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(details, file=sys.stderr)
        QMessageBox.critical(None, "程序异常", details[-3000:])

    sys.excepthook = report_exception
    app = QApplication(sys.argv)
    # 使用 Fusion 风格作为基底，避免 Windows 默认控件的样式干扰 QSS
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
