import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QFileDialog,
                               QTextEdit, QProgressBar, QLineEdit, QMessageBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal


class DetectionThread(QThread):
    update_frame = Signal(QImage)
    update_log = Signal(str)
    detection_complete = Signal()
    update_progress = Signal(int)

    def __init__(self, model_path, source_type, source_path=None):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type
        self.source_path = source_path
        self.model = None
        self.cap = None
        self.running = False
        self.confidence_threshold = 0.5

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def run(self):
        try:
            from ultralytics import YOLO
            self.update_log.emit(f"加载模型: {os.path.basename(self.model_path)}")
            self.model = YOLO(self.model_path)
            self.update_log.emit("模型加载完成")

            if self.source_type == 'camera':
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.update_log.emit("无法打开摄像头")
                    return
            elif self.source_type == 'video':
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.update_log.emit("无法打开视频")
                    return
            elif self.source_type == 'image':
                pass

            self.running = True
            frame_count = 0

            if self.source_type in ['camera', 'video']:
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.update_log.emit("读取结束")
                        break

                    self._process_frame(frame)

                    if self.source_type == 'video':
                        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        progress = int((frame_count / total_frames) * 100)
                        self.update_progress.emit(progress)
                        frame_count += 1

            elif self.source_type == 'image':
                frame = cv2.imread(self.source_path)
                self._process_frame(frame)
                self.update_progress.emit(100)

            self.detection_complete.emit()

        except Exception as e:
            self.update_log.emit(f"错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.running = False

    def _process_frame(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()

        detections = []
        for box in results[0].boxes:
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append(f"{cls_name} ({conf:.2f})")

        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.update_frame.emit(q_img)

        if detections:
            self.update_log.emit("检测到: " + ", ".join(detections))

    def stop(self):
        self.running = False
        self.wait()


class ObjectDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = ''
        self.detection_thread = None
        self.current_source_type = "camera"  # 默认摄像头
        self.current_source_path = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("YOLO检测系统")
        self.setGeometry(100, 100, 1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 控制区
        control_layout = QHBoxLayout()

        # 模型设置
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"))
        self.model_path_label = QLabel("未选择模型")
        self.browse_model_btn = QPushButton("浏览模型文件")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.browse_model_btn)
        control_layout.addLayout(model_layout)

        # 源选择（按钮形式）
        source_layout = QVBoxLayout()
        source_layout.addWidget(QLabel("输入源选择:"))

        # 源选择按钮组
        self.source_buttons = {}
        source_types = [("camera", "摄像头"), ("video", "视频文件"), ("image", "图像文件")]
        for typ, text in source_types:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, t=typ: self._set_source_type(t))
            self.source_buttons[typ] = btn
            source_layout.addWidget(btn)

        # 文件路径显示
        self.source_path_label = QLabel("未选择文件")
        self.browse_source_btn = QPushButton("选择文件")
        self.browse_source_btn.clicked.connect(self._browse_source)
        self.browse_source_btn.setEnabled(False)  # 默认禁用
        source_layout.addWidget(self.source_path_label)
        source_layout.addWidget(self.browse_source_btn)

        control_layout.addLayout(source_layout)

        # 检测控制
        det_layout = QVBoxLayout()

        # 置信度输入（替换滑块为输入框）
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence_input = QLineEdit("0.5")
        self.confidence_input.setMaximumWidth(60)
        conf_layout.addWidget(self.confidence_input)
        conf_layout.addWidget(QLabel("(0-1之间)"))

        # 开始/停止按钮
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self._start_detection)

        det_layout.addLayout(conf_layout)
        det_layout.addWidget(self.start_btn)
        control_layout.addLayout(det_layout)

        main_layout.addLayout(control_layout)

        # 显示区和结果区
        display_layout = QHBoxLayout()

        # 左侧显示区域
        self.display_label = QLabel("等待检测...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setStyleSheet("border: 1px solid #ccc;")  # 简单边框便于区分
        display_layout.addWidget(self.display_label)

        # 右侧结果区
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.progress_bar = QProgressBar()
        results_layout.addWidget(QLabel("检测日志:"))
        results_layout.addWidget(self.results_text)
        results_layout.addWidget(QLabel("进度:"))
        results_layout.addWidget(self.progress_bar)
        display_layout.addLayout(results_layout)

        main_layout.addLayout(display_layout)

    def _set_source_type(self, source_type):
        """设置源类型（按钮点击触发）"""
        self.current_source_type = source_type

        # 更新按钮状态（视觉区分）
        for typ, btn in self.source_buttons.items():
            btn.setChecked(typ == source_type)
            btn.setStyleSheet("background-color: lightblue;" if typ == source_type else "")

        # 启用/禁用文件选择按钮
        is_file_source = source_type in ["video", "image"]
        self.browse_source_btn.setEnabled(is_file_source)

        if is_file_source:
            self.source_path_label.setText("未选择文件")
            self.current_source_path = None
        else:
            self.source_path_label.setText("使用摄像头")

    def _browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "模型文件 (*.pt)")
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(os.path.basename(file_path))

    def _browse_source(self):
        """浏览源文件（视频或图像）"""
        if self.current_source_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        else:  # image
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.jpg *.png *.jpeg)")

        if file_path:
            self.current_source_path = file_path
            self.source_path_label.setText(os.path.basename(file_path))

    def _start_detection(self):
        """开始/停止检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            # 停止检测
            self.detection_thread.stop()
            self.start_btn.setText("开始检测")
            self.results_text.append("检测已停止")
            return

        # 验证设置
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件")
            return

        # 验证置信度输入
        try:
            confidence = float(self.confidence_input.text())
            if not (0 <= confidence <= 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "置信度必须是0-1之间的数字")
            return

        # 验证源
        if self.current_source_type in ["video", "image"] and not self.current_source_path:
            QMessageBox.warning(self, "警告", "请先选择文件")
            return

        # 启动检测线程
        self.detection_thread = DetectionThread(
            self.model_path, self.current_source_type, self.current_source_path
        )
        self.detection_thread.set_confidence_threshold(confidence)
        self.detection_thread.update_frame.connect(self._update_display)
        self.detection_thread.update_log.connect(self.results_text.append)
        self.detection_thread.detection_complete.connect(
            lambda: self.start_btn.setText("开始检测")
        )
        self.detection_thread.update_progress.connect(self.progress_bar.setValue)

        # 初始化状态
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setText("停止检测")
        self.detection_thread.start()

    def _update_display(self, q_img):
        """更新显示图像"""
        scaled_img = q_img.scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(QPixmap.fromImage(scaled_img))

    def closeEvent(self, event):
        """窗口关闭时停止线程"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionUI()
    window.show()
    sys.exit(app.exec())