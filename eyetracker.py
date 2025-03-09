#!/usr/bin/env python3
# v1.81 - UI updates and bug fixes with separate dialogs for pupil marking and ROI,
# using a polygon-based ROI selector, the original threshold slider for pupil threshold,
# and profile management using INI files.

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser  # for profile management
from datetime import datetime

# Update: use PySide6 instead of PyQt5.
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QDialog, QSlider,
    QInputDialog, QMessageBox, QFrame, QScrollArea, QProgressDialog,
    QCheckBox, QSizePolicy
)
from PySide6.QtGui import QAction, QPixmap, QImage, QDragEnterEvent, QDropEvent, QDesktopServices, QPainter, QPen
from PySide6.QtCore import Qt, QUrl, QStandardPaths, QSettings, QPoint, QTimer

# Multimedia classes for video playback
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from superqt import QRangeSlider

# Import the video processing module (circle detection version)
import video_processing
from results import ResultsView

MIN_PUPIL_RADIUS = 20

APP_CONFIG_PATH = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
if not os.path.exists(APP_CONFIG_PATH):
    os.makedirs(APP_CONFIG_PATH)
SETTINGS_FILE = os.path.join(APP_CONFIG_PATH, "settings.json")

# Directory for saving profile files (INI format)
PROFILE_DIR = os.path.join(APP_CONFIG_PATH, "profiles")
if not os.path.exists(PROFILE_DIR):
    os.makedirs(PROFILE_DIR)

def save_settings(output_directory):
    with open(SETTINGS_FILE, "w") as file:
        json.dump({"output_directory": output_directory}, file)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as file:
            return json.load(file).get("output_directory", "")
    return ""

#############################################
# Polygon ROI Selection Dialog
#############################################
class PolygonROIDialog(QDialog):
    """
    This dialog lets the user define an ROI as a polygon.
    The user clicks to add vertices (which are drawn on the image).
    Each vertex is clipped to remain within the image bounds.
    When the user clicks "Confirm ROI", the polygon is closed and
    the bounding rectangle of the polygon is returned as the ROI.
    """
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ROI (Polygon)")
        self.frame = frame.copy()
        self.original_height, self.original_width = frame.shape[:2]
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = int(screen.width() * 0.9)
        max_height = screen.height()
        self.scale_factor = min(1.0, max_width / self.original_width, max_height / self.original_height)
        self.vertices = []  # List of (x, y) in original image coordinates.
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        self.instruction_label = QLabel("Click to add vertices. When finished, click 'Confirm ROI'.", self)
        layout.addWidget(self.instruction_label)
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.display_frame()
        layout.addWidget(self.image_label)
        self.confirm_button = QPushButton("Confirm ROI", self)
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)
        self.image_label.mousePressEvent = self.add_vertex
    
    def display_frame(self):
        disp = self.frame.copy()
        if self.vertices:
            painter = QPainter()
            frame_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, w*ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            painter.begin(pixmap)
            pen = QPen(Qt.green, 2)
            painter.setPen(pen)
            for pt in self.vertices:
                painter.drawEllipse(QPoint(pt[0], pt[1]), 3, 3)
            for i in range(1, len(self.vertices)):
                painter.drawLine(QPoint(self.vertices[i-1][0], self.vertices[i-1][1]),
                                 QPoint(self.vertices[i][0], self.vertices[i][1]))
            painter.end()
            disp = pixmap.toImage()
        else:
            disp = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = disp.shape
            disp = QImage(disp.data, w, h, w*ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(disp)
        pixmap = pixmap.scaled(int(self.original_width * self.scale_factor),
                                int(self.original_height * self.scale_factor),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    
    def add_vertex(self, event):
        pos = event.pos()
        disp_size = self.image_label.pixmap().size()
        scale_x = self.original_width / disp_size.width()
        scale_y = self.original_height / disp_size.height()
        x = int(pos.x() * scale_x)
        y = int(pos.y() * scale_y)
        # Clip to image boundaries.
        x = max(0, min(x, self.original_width - 1))
        y = max(0, min(y, self.original_height - 1))
        self.vertices.append((x, y))
        self.display_frame()
    
    def getPolygonROI(self):
        if len(self.vertices) < 3:
            return None
        pts = np.array(self.vertices)
        x, y, w, h = cv2.boundingRect(pts)
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.original_width - x)
        h = min(h, self.original_height - y)
        return (x, y, w, h)

#############################################
# Pupil Center Dialog (line drawing)
#############################################
class PupilCenterDialog(QDialog):
    """
    This dialog instructs the user to drag a line over the longest part of the pupil.
    The midpoint of that line is taken as the pupil center and half the line's length as the initial pupil radius.
    """
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mark Pupil")
        self.frame = frame.copy()
        self.orig_h, self.orig_w = self.frame.shape[:2]
        screen = QApplication.primaryScreen().availableGeometry()
        max_w = int(screen.width() * 0.9)
        max_h = screen.height()
        self.scale_factor = min(1.0, max_w / self.orig_w, max_h / self.orig_h)
        self.start_point = None
        self.end_point = None
        self.pupil_center = None
        self.initial_radius = None
        self.initUI()
    
    def initUI(self):
        self.layout = QVBoxLayout()
        self.instruction_label = QLabel("Drag a line over the longest part of the pupil.", self)
        self.layout.addWidget(self.instruction_label)
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.display_frame()
        self.layout.addWidget(self.image_label)
        self.confirm_button = QPushButton("Confirm Pupil", self)
        self.confirm_button.clicked.connect(self.accept)
        self.layout.addWidget(self.confirm_button)
        self.setLayout(self.layout)
        self.image_label.mousePressEvent = self.line_start
        self.image_label.mouseReleaseEvent = self.line_end
    
    def display_frame(self):
        disp = self.frame.copy()
        if self.start_point is not None and self.end_point is not None:
            cv2.line(disp, self.start_point, self.end_point, (0,255,0), 2)
        frame_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, w*ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(int(w * self.scale_factor), int(h * self.scale_factor),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    
    def line_start(self, event):
        pos = event.pos()
        disp_size = self.image_label.pixmap().size()
        scale_x = self.orig_w / disp_size.width()
        scale_y = self.orig_h / disp_size.height()
        self.start_point = (int(pos.x() * scale_x), int(pos.y() * scale_y))
        self.end_point = self.start_point
        self.display_frame()
    
    def line_end(self, event):
        pos = event.pos()
        disp_size = self.image_label.pixmap().size()
        scale_x = self.orig_w / disp_size.width()
        scale_y = self.orig_h / disp_size.height()
        self.end_point = (int(pos.x() * scale_x), int(pos.y() * scale_y))
        self.pupil_center = (int((self.start_point[0] + self.end_point[0]) / 2),
                             int((self.start_point[1] + self.end_point[1]) / 2))
        line_length = np.linalg.norm(np.array(self.end_point) - np.array(self.start_point))
        self.initial_radius = line_length / 2.0
        print(f"Pupil marked at: {self.pupil_center}, radius: {self.initial_radius:.2f}")
        self.display_frame()
    
    def getPupilCenterAndRadius(self):
        return self.pupil_center, self.initial_radius

#############################################
# Main Application Window
#############################################
class EyeTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Application")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setMaximumSize(screen.width(), screen.height())
        self.max_video_width = int(screen.width() * 0.9)
        self.create_menus()
        
        self.setGeometry(100, 100, 1200, 800)
        self.setAcceptDrops(True)
        
        self.video_path = None
        self.output_directory = load_settings()
        self.video_capture = None
        self.frame = None
        self.fps = 0
        self.roi = None  # ROI from PolygonROIDialog (bounding rectangle)
        self.initial_cumulative_center = None  # from PupilCenterDialog
        self.initial_pupil_radius = None         # from PupilCenterDialog
        self.pupil_threshold = 50  # use slider default
        self.current_session_dir = None
        self.tracking_video_path = None
        self.tracking_csv_path = None
        self.rotation_angle = 0
        
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout()

        # Create a widget for the top controls with minimal spacing.
        top_controls = QWidget()
        top_layout = QVBoxLayout(top_controls)
        top_layout.setContentsMargins(5, 5, 5, 5)
        top_layout.setSpacing(5)

        # Row 1: Video controls
        video_layout = QHBoxLayout()
        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_button)
        self.video_label = QLabel("No video selected", self)
        video_layout.addWidget(self.video_label)
        self.rotate_video_button = QPushButton("Rotate 90°", self)
        natural_size = self.rotate_video_button.sizeHint()
        self.rotate_video_button.setFixedSize(natural_size)
        self.rotate_video_button.clicked.connect(self.rotate_video)
        video_layout.addWidget(self.rotate_video_button)
        top_layout.addLayout(video_layout)

        # Row 2: Output directory controls
        directory_layout = QHBoxLayout()
        self.save_directory_button = QPushButton("Select Output Directory", self)
        self.save_directory_button.clicked.connect(self.select_output_directory)
        directory_layout.addWidget(self.save_directory_button)
        self.directory_label = QLabel(f"Output Directory: {self.output_directory if self.output_directory else 'Not selected'}", self)
        directory_layout.addWidget(self.directory_label)
        self.show_results_button = QPushButton("Show Results Files", self)
        self.show_results_button.clicked.connect(self.show_results_files)
        directory_layout.addWidget(self.show_results_button)
        top_layout.addLayout(directory_layout)

        # Row 3: Threshold slider row
        threshold_layout = QHBoxLayout()
        threshold_text = QLabel("Pupil Threshold:", self)
        threshold_layout.addWidget(threshold_text)
        self.threshold_slider = QSlider(Qt.Horizontal, self)
        self.threshold_slider.setMinimum(30)
        self.threshold_slider.setMaximum(150)
        self.threshold_slider.setValue(self.pupil_threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel(f"{self.pupil_threshold}", self)
        threshold_layout.addWidget(self.threshold_label)
        top_layout.addLayout(threshold_layout)

        # Row 4: Time range slider row
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(100)
        self.range_slider.setValue((0, 100))
        self.range_slider.valueChanged.connect(self.update_time_range)
        slider_layout = QHBoxLayout()
        self.start_time_label = QLabel("0:00", self)
        slider_layout.addWidget(self.start_time_label)
        slider_layout.addWidget(self.range_slider)
        self.end_time_label = QLabel("0:00", self)
        slider_layout.addWidget(self.end_time_label)
        top_layout.addLayout(slider_layout)
        self.range_slider.setVisible(False)
        self.start_time_label.setVisible(False)
        self.end_time_label.setVisible(False)

        # Add the top_controls widget at the top.
        main_layout.addWidget(top_controls)

        # Add a vertical spacer to push the remaining content down.
        main_layout.addStretch(1)

        # Add remaining UI elements below the spacer.
        # Video display.
        self.image_label = QLabel(self)
        self.image_label.setFixedHeight(20)
        main_layout.addWidget(self.image_label)
        
        # Contrast and Exposure sliders in one row.
        ce_layout = QHBoxLayout()
        self.contrast_label = QLabel("Contrast:", self)
        ce_layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        ce_layout.addWidget(self.contrast_slider)
        self.exposure_label = QLabel("Exposure:", self)
        ce_layout.addWidget(self.exposure_label)
        self.exposure_slider = QSlider(Qt.Horizontal, self)
        self.exposure_slider.setMinimum(-50)
        self.exposure_slider.setMaximum(50)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self.update_exposure)
        ce_layout.addWidget(self.exposure_slider)
        main_layout.addLayout(ce_layout)
        self.contrast_slider.setVisible(False)
        self.contrast_label.setVisible(False)
        self.exposure_slider.setVisible(False)
        self.exposure_label.setVisible(False)
        
        # Control buttons row.
        control_layout = QHBoxLayout()
        self.mark_pupil_button = QPushButton("Mark Pupil", self)
        self.mark_pupil_button.clicked.connect(self.mark_pupil_center)
        control_layout.addWidget(self.mark_pupil_button)
        self.select_roi_button = QPushButton("Select Eye Region", self)
        self.select_roi_button.clicked.connect(self.select_polygon_roi)
        self.select_roi_button.setEnabled(False)
        control_layout.addWidget(self.select_roi_button)
        self.start_tracking_button = QPushButton("Start Tracking", self)
        self.start_tracking_button.setEnabled(False)
        self.start_tracking_button.clicked.connect(self.start_tracking)
        control_layout.addWidget(self.start_tracking_button)
        self.view_results_button = QPushButton("View Results", self)
        self.view_results_button.clicked.connect(self.view_results)
        self.view_results_button.setEnabled(False)
        control_layout.addWidget(self.view_results_button)
        main_layout.addLayout(control_layout)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(divider)
        
        # Reset, Profile, and load results buttons
        reset_layout = QHBoxLayout()
        self.save_profile_button = QPushButton("Save Profile", self)
        self.save_profile_button.clicked.connect(self.save_profile)
        reset_layout.addWidget(self.save_profile_button)
        self.load_profile_button = QPushButton("Load Profile", self)
        self.load_profile_button.clicked.connect(self.load_profile)
        reset_layout.addWidget(self.load_profile_button)
        self.restart_button = QPushButton("Reset", self)
        self.restart_button.clicked.connect(self.restart_analysis)
        reset_layout.addWidget(self.restart_button)
        self.load_saved_results_button = QPushButton("Load Results", self)
        self.load_saved_results_button.clicked.connect(self.load_saved_results)
        reset_layout.addWidget(self.load_saved_results_button)
        main_layout.addLayout(reset_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

    def create_menus(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")
        load_video_action = QAction("Load Video", self)
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)

        show_results_files_action = QAction("Show Results Files", self)
        show_results_files_action.triggered.connect(self.show_results_files)
        file_menu.addAction(show_results_files_action)

        load_results_action = QAction("Load Results", self)
        load_results_action.triggered.connect(self.load_saved_results)
        file_menu.addAction(load_results_action)

        # Profiles Menu
        profiles_menu = menubar.addMenu("Profiles")
        save_profile_action = QAction("Save Profile", self)
        save_profile_action.triggered.connect(self.save_profile)
        profiles_menu.addAction(save_profile_action)

        load_profile_action = QAction("Load Profile", self)
        load_profile_action.triggered.connect(self.load_profile)
        profiles_menu.addAction(load_profile_action)

        delete_profile_action = QAction("Delete Profile", self)
        delete_profile_action.triggered.connect(self.delete_profile)
        profiles_menu.addAction(delete_profile_action)

        # Help Menu
        help_menu = menubar.addMenu("Help")
        website_action = QAction("Visit Website", self)
        website_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://robbieswebsites.com")))
        help_menu.addAction(website_action)

    def delete_profile(self):
        import os
        from PySide6.QtWidgets import QMessageBox, QInputDialog
        # Get list of profile INI files from PROFILE_DIR (assumed to be a global variable or attribute)
        profiles = [f for f in os.listdir(PROFILE_DIR) if f.endswith(".ini")]
        if not profiles:
            QMessageBox.information(self, "No Profiles", "No profiles available to delete.")
            return
        profile, ok = QInputDialog.getItem(self, "Delete Profile", "Select profile to delete:", profiles, 0, False)
        if ok and profile:
            reply = QMessageBox.question(self, "Confirm Deletion",
                                        f"Are you sure you want to delete the profile '{profile}'?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                os.remove(os.path.join(PROFILE_DIR, profile))
                QMessageBox.information(self, "Profile Deleted", f"Profile '{profile}' has been deleted.")
    
    def save_profile(self):
        # Save current slider settings (pupil_threshold, contrast, exposure) to a new profile.
        name, ok = QInputDialog.getText(self, "Save Profile", "Enter profile name:")
        if not ok or not name:
            return
        config = configparser.ConfigParser()
        config[name] = {
            "pupil_threshold": str(self.pupil_threshold),
            "contrast": str(self.contrast_slider.value()),
            "exposure": str(self.exposure_slider.value())
        }
        profile_path = os.path.join(PROFILE_DIR, f"{name}.ini")
        with open(profile_path, "w") as configfile:
            config.write(configfile)
        print(f"Profile '{name}' saved.")
    
    def load_profile(self):
        # Let the user choose a profile from the PROFILE_DIR and load its settings.
        profiles = [f for f in os.listdir(PROFILE_DIR) if f.endswith(".ini")]
        if not profiles:
            QMessageBox.information(self, "No Profiles", "No profiles found.")
            return
        name, ok = QInputDialog.getItem(self, "Load Profile", "Select profile:", profiles, 0, False)
        if not ok:
            return
        profile_path = os.path.join(PROFILE_DIR, name)
        config = configparser.ConfigParser()
        config.read(profile_path)
        section = config.sections()[0]
        self.pupil_threshold = int(config[section]["pupil_threshold"])
        self.threshold_slider.setValue(self.pupil_threshold)
        self.contrast_slider.setValue(int(config[section]["contrast"]))
        self.exposure_slider.setValue(int(config[section]["exposure"]))
        self.threshold_label.setText(f"Pupil Threshold: {self.pupil_threshold}")
        print(f"Profile '{section}' loaded.")
    
    def mark_pupil_center(self):
        if self.frame is None:
            QMessageBox.information(self, "No Frame", "Please load a video first.")
            return
        dialog = PupilCenterDialog(video_processing.apply_rotation(self.frame.copy(), self.rotation_angle), self)
        if dialog.exec():
            center, radius = dialog.getPupilCenterAndRadius()
            if center is not None and radius is not None:
                self.initial_cumulative_center = center
                self.initial_pupil_radius = radius
                print(f"Initial pupil marked: {self.initial_cumulative_center}, radius: {self.initial_pupil_radius:.2f}")
                self.select_roi_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Selection Error", "No pupil was marked.")
    
    def select_polygon_roi(self):
        if not self.output_directory:
            QMessageBox.warning(self, "Output Directory Required", "Please select an output directory first.")
            return
        if self.frame is not None:
            dialog = PolygonROIDialog(video_processing.apply_rotation(self.frame.copy(), self.rotation_angle), self)
            if dialog.exec():
                poly_roi = dialog.getPolygonROI()
                print(f"DEBUG: Returned Polygon ROI (bounding rect): {poly_roi}")
                if poly_roi and poly_roi[2] > 0 and poly_roi[3] > 0:
                    self.roi = poly_roi
                    print(f"Polygon ROI Confirmed: {self.roi}")
                    if self.initial_cumulative_center is not None:
                        self.start_tracking_button.setEnabled(True)
                    else:
                        QMessageBox.warning(self, "Missing Pupil", "Please mark the pupil first.")
                else:
                    QMessageBox.warning(self, "Selection Error", "Please mark a valid polygon ROI.")
    
    def rotate_video(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        print(f"Video rotated to {self.rotation_angle} degrees")
        if self.frame is not None:
            self.display_frame(self.frame)
    
    def update_threshold(self, value):
        self.pupil_threshold = value
        self.threshold_label.setText(f"{value}")
    
    def show_results_files(self):
        if not self.output_directory:
            QMessageBox.warning(self, "Output Directory Required", "Please select an output directory first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_directory))
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.video_path = file_path
                self.video_capture = cv2.VideoCapture(file_path)
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                self.range_slider.setMinimum(0)
                self.range_slider.setMaximum(total_frames - 1)
                self.range_slider.setValue((0, total_frames - 1))
                self.video_label.setText(f"Selected Video: {os.path.basename(file_path)}")
                self.display_frame_at_time(0)
                self.update_time_range()
                break
    
    def load_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                   "Video Files (*.mp4 *.avi *.mov *.mkv)", options=options)
        if file_path:
            self.video_path = file_path
            self.video_capture = cv2.VideoCapture(file_path)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.range_slider.setMinimum(0)
            self.range_slider.setMaximum(total_frames - 1)
            self.range_slider.setValue((0, total_frames - 1))
            self.video_label.setText(f"Selected Video: {os.path.basename(file_path)}")
            self.display_frame_at_time(0)
            self.update_time_range()
            
            video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_width = self.width()
            extra = 300
            screen = QApplication.primaryScreen().availableGeometry()
            new_width = current_width if video_width < current_width else min(video_width, screen.width())
            new_height = min(video_height + extra, screen.height())
            self.resize(new_width, new_height)
            self.contrast_slider.setVisible(True)
            self.contrast_label.setVisible(True)
            self.exposure_slider.setVisible(True)
            self.exposure_label.setVisible(True)
            self.range_slider.setVisible(True)
            self.start_time_label.setVisible(True)
            self.end_time_label.setVisible(True)
    
    def display_frame_at_time(self, frame_number):
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_capture.read()
            if ret:
                self.frame = frame.copy()
                self.display_frame(self.frame)
    
    def display_frame(self, frame):
        contrast = self.contrast_slider.value() / 100.0 if self.contrast_slider.isVisible() else 1.0
        exposure = self.exposure_slider.value() if self.exposure_slider.isVisible() else 0
        frame_adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=exposure)
        frame_to_show = video_processing.apply_rotation(frame_adjusted.copy(), self.rotation_angle)
        frame_to_show = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_to_show.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame_to_show.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.width() > self.max_video_width:
            pixmap = pixmap.scaledToWidth(self.max_video_width, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedHeight(0)
        self.image_label.setMinimumHeight(0)
        self.image_label.setMaximumHeight(16777215)
    
    def update_contrast(self, value):
        if self.frame is not None:
            self.display_frame(self.frame)
    
    def update_exposure(self, value):
        if self.frame is not None:
            self.display_frame(self.frame)
    
    def select_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_directory = dir_path
            self.directory_label.setText(f"Output Directory: {self.output_directory}")
            save_settings(self.output_directory)
    
    def update_time_range(self):
        start_frame, end_frame = self.range_slider.value()
        self.update_time_label(self.start_time_label, start_frame)
        self.update_time_label(self.end_time_label, end_frame)
        self.display_frame_at_time(start_frame)
    
    def update_time_label(self, label, frame_number):
        if self.fps > 0:
            seconds = frame_number / self.fps
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            label.setText(f"{minutes}:{seconds:02d}")
    
    def start_tracking(self):
        if not self.output_directory:
            QMessageBox.warning(self, "Output Directory Required", "Please select an output directory first.")
            return
        if self.video_path and self.output_directory and self.roi and self.initial_cumulative_center:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session_dir = os.path.join(self.output_directory, f"tracking_{timestamp}")
            os.makedirs(self.current_session_dir, exist_ok=True)
            print(f"Starting pupil tracking... Outputs will be saved to: {self.current_session_dir}")
            
            progress_dialog = QProgressDialog("Processing, please wait...", "Cancel", 0, 100, self)
            progress_dialog.setWindowModality(Qt.ApplicationModal)
            progress_dialog.setWindowTitle("Processing")
            progress_dialog.show()
            
            contrast = self.contrast_slider.value() / 100.0 if self.contrast_slider.isVisible() else 1.0
            exposure = self.exposure_slider.value() if self.exposure_slider.isVisible() else 0
            
            output_path, positions = video_processing.track_pupil_with_video_output(
                video_capture=self.video_capture,
                roi=self.roi,
                threshold_value=self.pupil_threshold,
                rotation_angle=self.rotation_angle,
                fps=self.fps,
                range_slider_value=self.range_slider.value(),
                current_session_dir=self.current_session_dir,
                contrast=contrast,
                exposure=exposure,
                progress_callback=lambda p: progress_dialog.setValue(p),
                initial_cumulative_center=self.initial_cumulative_center,
                initial_pupil_radius=self.initial_pupil_radius
            )
            progress_dialog.close()
            self.tracking_video_path = output_path
            self.tracking_csv_path = video_processing.save_tracking_data(positions, self.current_session_dir)
            if self.tracking_csv_path is None:
                QMessageBox.warning(self, "No Tracking Data", "No pupil tracking data was generated. Please check your ROI and threshold settings.")
                return
            df = pd.read_csv(self.tracking_csv_path)
            video_processing.plot_normalized_tracking_data(df, self.current_session_dir)
            print(f"Tracking video saved to: {output_path}")
            self.view_results_button.setEnabled(True)
    
    def view_results(self):
        if not self.output_directory:
            QMessageBox.warning(self, "Output Directory Required", "Please select an output directory first.")
            return
        if self.tracking_video_path and self.tracking_csv_path:
            self.results_view = ResultsView(self.tracking_video_path, self.tracking_csv_path)
            # Ensure the results view is a top-level window.
            self.results_view.setParent(None)
            self.results_view.setWindowFlags(Qt.Window)
            self.results_view.show()
            self.results_view.raise_()
            self.results_view.activateWindow()
            QApplication.processEvents()
            # For macOS, force the app to become frontmost via AppleScript.
            if sys.platform == "darwin":
                try:
                    import subprocess
                    subprocess.call([
                        "osascript",
                        "-e",
                        'tell application "System Events" to set frontmost of the first process whose unix id is {} to true'.format(os.getpid())
                    ])
                except Exception as e:
                    print("Error forcing activation:", e)
    
    def restart_analysis(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_path = None
        self.frame = None
        self.fps = 0
        self.roi = None
        self.initial_cumulative_center = None
        self.initial_pupil_radius = None
        self.current_session_dir = None
        self.tracking_video_path = None
        self.tracking_csv_path = None
        
        self.video_label.setText("No video selected")
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))
        self.start_time_label.setText("0:00")
        self.end_time_label.setText("0:00")
        self.image_label.clear()
        self.start_tracking_button.setEnabled(False)
        self.view_results_button.setEnabled(False)
        
        self.load_video()
    
    def load_saved_results(self):
        start_dir = self.output_directory if self.output_directory else ""
        directory = QFileDialog.getExistingDirectory(self, "Select Results Folder", start_dir)
        if not directory:
            return

        csv_candidates = [f for f in os.listdir(directory) if f.lower().endswith(".csv") and "eye_tracking_output" in f]
        if not csv_candidates:
            QMessageBox.information(self, "Error", "No results CSV file found in the selected folder.")
            return
        elif len(csv_candidates) == 1:
            csv_file = os.path.join(directory, csv_candidates[0])
        else:
            item, ok = QInputDialog.getItem(self, "Select CSV File", "Multiple CSV files found. Please select one:", csv_candidates, 0, False)
            if ok:
                csv_file = os.path.join(directory, item)
            else:
                return

        video_candidates = [f for f in os.listdir(directory) if f.lower().endswith((".avi", ".mov", ".mp4"))]
        if not video_candidates:
            QMessageBox.information(self, "Error", "No tracking video file found in the selected folder.")
            return
        elif len(video_candidates) == 1:
            video_file = os.path.join(directory, video_candidates[0])
        else:
            item, ok = QInputDialog.getItem(self, "Select Video File", "Multiple video files found. Please select one:", video_candidates, 0, False)
            if ok:
                video_file = os.path.join(directory, item)
            else:
                return

        self.results_view = ResultsView(video_file, csv_file)
        # Ensure it's a top-level window.
        self.results_view.setParent(None)
        self.results_view.setWindowFlags(Qt.Window)
        self.results_view.show()
        self.results_view.raise_()
        self.results_view.activateWindow()
        QApplication.processEvents()

        # For macOS, force the app to become frontmost via AppleScript.
        if sys.platform == "darwin":
            try:
                import subprocess
                subprocess.call([
                    "osascript",
                    "-e",
                    'tell application "System Events" to set frontmost of the first process whose unix id is {} to true'.format(os.getpid())
                ])
            except Exception as e:
                print("Error forcing activation:", e)

def show_startup_warning():
    settings = QSettings("YourOrganization", "YourAppName")
    if settings.value("dont_show_warning", False, type=bool):
        return
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Disclaimer")
    msg_box.setText("Designed for research and education. Not suitable for medical diagnosis.")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.button(QMessageBox.Ok).setText("I Understand")
    checkbox = QCheckBox("Understood, do not show again")
    msg_box.setCheckBox(checkbox)
    msg_box.exec()
    if checkbox.isChecked():
        settings.setValue("dont_show_warning", True)

# Define the path to the default profile.
DEFAULT_PROFILE_PATH = os.path.join(PROFILE_DIR, "default.ini")

def ensure_default_profile():
    config = configparser.ConfigParser()
    # If the default profile doesn't exist, create one.
    if not os.path.exists(DEFAULT_PROFILE_PATH):
        config["Default"] = {
            "pupil_threshold": "50",
            "contrast": "100",
            "exposure": "0"
        }
        with open(DEFAULT_PROFILE_PATH, "w") as configfile:
            config.write(configfile)
        print("Default profile created.")
    else:
        print("Default profile exists.")


def main():
    app = QApplication(sys.argv)
    def resource_path(relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    app.setWindowIcon(QtGui.QIcon(resource_path("EyeballTarget.png")))
    show_startup_warning()
    ensure_default_profile()
    window = EyeTrackingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()