#!/usr/bin/env python3
import sys
import os
import pandas as pd
import vlc
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QApplication, QHBoxLayout, 
                               QComboBox, QPushButton, QSlider, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class ResultsView(QWidget):
    def __init__(self, video_path, csv_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Results View")
        self.video_path = video_path
        self.csv_path = csv_path
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # --- Video Playback Section using Python-VLC ---
        self.video_frame = QWidget(self)
        self.video_frame.setAttribute(Qt.WA_NativeWindow, True)
        self.video_frame.setMinimumSize(640, 360)
        layout.addWidget(self.video_frame)
        
        # Create VLC instance and media player.
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()
        media = self.vlc_instance.media_new(self.video_path)
        self.vlc_player.set_media(media)
        
        # Set video output based on platform.
        if sys.platform.startswith("linux"):
            self.vlc_player.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.vlc_player.set_hwnd(self.video_frame.winId())
        elif sys.platform == "darwin":
            self.vlc_player.set_nsobject(int(self.video_frame.winId()))
        
        # Create a seek slider with fixed height.
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setFixedHeight(30)
        self.seek_slider.setRange(0, 100)
        layout.addWidget(self.seek_slider)
        
        # Timer to update the slider from the player's current time.
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.start(100)
        
        # Playback controls.
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        controls_layout.addStretch()
        # Playback speed control.
        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.1x", "0.25x", "0.5x", "1.0x"])
        self.speed_combo.setCurrentIndex(3)  # 1.0x
        self.speed_combo.currentIndexChanged.connect(self.update_playback_rate)
        controls_layout.addWidget(self.speed_combo)
        layout.addLayout(controls_layout)
        
        # Connect slider events to seeking.
        self.seek_slider.sliderReleased.connect(self.seek_video)
        self.seek_slider.sliderMoved.connect(self.seek_video)
        
        # --- Graph Section ---
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.df = pd.read_csv(self.csv_path)
        self.ax1.plot(self.df['Time (ms)'], self.df['X Deviation'], 'b-')
        self.ax2.plot(self.df['Time (ms)'], self.df['Y Deviation'], 'g-')
        self.ax1.set_ylabel("X Dev")
        self.ax2.set_ylabel("Y Dev")
        self.ax2.set_xlabel("Time (ms)")
        
        t_min = self.df['Time (ms)'].min()
        t_max = self.df['Time (ms)'].max()
        self.ax1.set_xlim(t_min, t_max)
        self.ax2.set_xlim(t_min, t_max)
        
        cur_ylim1 = self.ax1.get_ylim()
        self.ax1.set_ylim(min(cur_ylim1[0], -40), max(cur_ylim1[1], 40))
        cur_ylim2 = self.ax2.get_ylim()
        self.ax2.set_ylim(min(cur_ylim2[0], -20), max(cur_ylim2[1], 20))
        
        self.vline1 = self.ax1.axvline(x=0, color='r', linewidth=1)
        self.vline2 = self.ax2.axvline(x=0, color='r', linewidth=1)
        
        # Timer to update graph marker lines.
        self.graph_timer = QTimer(self)
        self.graph_timer.timeout.connect(self.on_timer_timeout)
        self.graph_timer.start(10)
    
    def toggle_play(self):
        if self.vlc_player.is_playing():
            self.vlc_player.pause()
            self.play_button.setText("Play")
        else:
            video_length = self.vlc_player.get_length()
            current_time = self.vlc_player.get_time()
            if video_length > 0 and current_time >= video_length - 100:
                self.vlc_player.set_time(0)
            self.vlc_player.play()
            self.play_button.setText("Pause")
    
    def update_position(self):
        current_time = self.vlc_player.get_time()
        video_length = self.vlc_player.get_length()
        if video_length > 0 and self.seek_slider.maximum() < video_length:
            self.seek_slider.setRange(0, video_length)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(current_time)
        self.seek_slider.blockSignals(False)
    
    def seek_video(self):
        position = self.seek_slider.value()
        self.vlc_player.set_time(position)
    
    def update_playback_rate(self, index):
        mapping = {0: 0.1, 1: 0.25, 2: 0.5, 3: 1.0}
        self.vlc_player.set_rate(mapping.get(index, 1.0))
    
    def on_timer_timeout(self):
        current_time = self.vlc_player.get_time()
        new_xdata = [current_time, current_time]
        self.vline1.set_xdata(new_xdata)
        self.vline2.set_xdata(new_xdata)
        self.canvas.draw_idle()
    
    def closeEvent(self, event):
        self.graph_timer.stop()
        self.position_timer.stop()
        self.vlc_player.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    from PySide6.QtCore import QTimer
    app = QApplication(sys.argv)
    video_path = "sample_video.mp4"      # update as needed
    csv_path = "eye_tracking_output.csv"   # update as needed
    window = ResultsView(video_path, csv_path)
    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    window.show()
    QTimer.singleShot(1000, lambda: (
        window.setWindowFlag(Qt.WindowStaysOnTopHint, False),
        window.showNormal(),
        window.raise_(),
        window.activateWindow()
    ))
    sys.exit(app.exec())