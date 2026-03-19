import datetime
import json
import os
import shutil
import threading
import wave
from pathlib import Path

import pyaudio
import torch
import warnings

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

warnings.filterwarnings("ignore")

# Desktop simulation for mobile layout
Window.size = (360, 640)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

BASE_DIR = Path(__file__).resolve().parent
TEMP_FILE = BASE_DIR / "ambient_temp_mobile.wav"
FLAGGED_DIR = BASE_DIR / "flagged_calls_mobile"
MODEL_PATH = BASE_DIR / "model.pth"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"

ALERT_FLOOR_THRESHOLD = 0.55
CONSECUTIVE_ALERTS_REQUIRED = 2
MONITOR_SENSITIVITY = 0.72
MONITOR_MODEL_WEIGHT = 0.76
MONITOR_ARTIFACT_WEIGHT = 0.24
MONITOR_CHUNK_SECONDS = 0.9
MONITOR_HOP_SECONDS = 0.35

FLAGGED_DIR.mkdir(parents=True, exist_ok=True)


def load_base_threshold(default: float = 0.5) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", default))
        return threshold if 0.1 <= threshold <= 0.9 else default
    except Exception:
        return default


class VachaShieldApp(App):
    def build(self):
        self.is_monitoring = False
        self.model = None
        self.base_threshold = load_base_threshold(0.5)
        self.alert_streak = 0
        self.cycle_count = 0

        layout = BoxLayout(orientation="vertical", padding=20, spacing=16)

        with layout.canvas.before:
            Color(0.05, 0.08, 0.12, 1)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        self.title_label = Label(
            text="[b]VACHA-SHIELD[/b]\nMobile Call Guard",
            markup=True,
            font_size="24sp",
            color=(0.1, 0.9, 1, 1),
            size_hint=(1, 0.18),
            halign="center",
        )
        layout.add_widget(self.title_label)

        self.status_label = Label(
            text="SYSTEM OFFLINE\nReady",
            font_size="16sp",
            color=(0.7, 0.7, 0.7, 1),
            size_hint=(1, 0.12),
            halign="center",
        )
        layout.add_widget(self.status_label)

        self.analysis_label = Label(
            text="No active analysis.",
            font_size="15sp",
            color=(1, 1, 1, 1),
            size_hint=(1, 0.45),
            halign="center",
            valign="middle",
        )
        self.analysis_label.bind(size=self.analysis_label.setter("text_size"))
        layout.add_widget(self.analysis_label)

        self.call_btn = Button(
            text="START BACKGROUND CALL SCAN",
            font_size="16sp",
            background_color=(0.1, 0.7, 0.5, 1),
            size_hint=(1, 0.16),
        )
        self.call_btn.bind(on_press=self.toggle_monitoring)
        layout.add_widget(self.call_btn)

        threading.Thread(target=self.init_ml_engine, daemon=True).start()
        return layout

    def _update_rect(self, instance, _value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def init_ml_engine(self):
        Clock.schedule_once(lambda _dt: self.update_status("Loading AI core..."), 0)
        try:
            if not MODEL_PATH.exists():
                Clock.schedule_once(lambda _dt: self.update_status("ERROR: model.pth missing", (1, 0.2, 0.2, 1)), 0)
                return

            model = AudioCNN(num_classes=1)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            self.model = model

            Clock.schedule_once(
                lambda _dt: self.update_status(
                    f"SYSTEM ONLINE\nThr={max(self.base_threshold, ALERT_FLOOR_THRESHOLD):.2f}",
                    (0.5, 1, 0.5, 1),
                ),
                0,
            )
        except Exception as exc:
            Clock.schedule_once(lambda _dt: self.update_status(f"Engine error: {str(exc)[:30]}", (1, 0.2, 0.2, 1)), 0)

    def update_status(self, text, color=(0.7, 0.7, 0.7, 1)):
        self.status_label.text = text
        self.status_label.color = color

    def update_analysis(self, text, color=(1, 1, 1, 1)):
        self.analysis_label.text = text
        self.analysis_label.color = color

    def toggle_monitoring(self, _instance):
        if self.model is None:
            self.update_analysis("AI engine still loading...", (1, 0.4, 0.4, 1))
            return

        if not self.is_monitoring:
            self.is_monitoring = True
            self.alert_streak = 0
            self.cycle_count = 0
            self.call_btn.text = "STOP SCAN"
            self.call_btn.background_color = (0.9, 0.2, 0.2, 1)
            self.update_status("CALL ACTIVE\nMonitoring...", (0.1, 0.9, 1, 1))
            threading.Thread(target=self.monitoring_loop, daemon=True).start()
        else:
            self.is_monitoring = False
            self.call_btn.text = "START BACKGROUND CALL SCAN"
            self.call_btn.background_color = (0.1, 0.7, 0.5, 1)
            self.update_status("SYSTEM ONLINE\nReady", (0.5, 1, 0.5, 1))
            self.update_analysis("Monitoring stopped.")

    def record_chunk(self) -> bool:
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )

            frames = []
            total_frames = int(SAMPLE_RATE / FRAMES_PER_BUFFER * CHUNK_SECONDS)
            for _ in range(total_frames):
                if not self.is_monitoring:
                    break
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            if not frames:
                return False

            with wave.open(str(TEMP_FILE), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            return True
        except Exception as exc:
            Clock.schedule_once(lambda _dt: self.update_analysis(f"Mic error: {str(exc)[:50]}", (1, 0.2, 0.2, 1)), 0)
            return False
        finally:
            p.terminate()

    def monitoring_loop(self):
        while self.is_monitoring:
            if not self.record_chunk():
                continue

            self.cycle_count += 1

            try:
                result = predict_deepfake_from_file(
                    audio_path=str(TEMP_FILE),
                    model=self.model,
                    device=DEVICE,
                    threshold=self.base_threshold,
                    chunk_seconds=MONITOR_CHUNK_SECONDS,
                    hop_seconds=MONITOR_HOP_SECONDS,
                    sensitivity=MONITOR_SENSITIVITY,
                    model_weight=MONITOR_MODEL_WEIGHT,
                    artifact_weight=MONITOR_ARTIFACT_WEIGHT,
                )

                synthetic = float(result["synthetic_probability"])
                human = float(result["human_probability"])
                threshold = max(float(result.get("threshold", self.base_threshold)), ALERT_FLOOR_THRESHOLD)

                if synthetic > threshold:
                    self.alert_streak += 1
                else:
                    self.alert_streak = 0

                if self.alert_streak >= CONSECUTIVE_ALERTS_REQUIRED:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    destination = FLAGGED_DIR / f"mobile_alert_{timestamp}_prob{int(synthetic * 100)}.wav"
                    shutil.copy(str(TEMP_FILE), str(destination))
                    text = (
                        "[b]POSSIBLE AI CLONE ALERT[/b]\n"
                        f"AI: {synthetic * 100:.1f}% | Human: {human * 100:.1f}%\n"
                        f"Threshold: {threshold * 100:.1f}%\n"
                        f"Saved: {destination.name}"
                    )
                    color = (1, 0.25, 0.25, 1)
                    self.alert_streak = 0
                else:
                    text = (
                        f"Cycle {self.cycle_count}\n"
                        f"Human: {human * 100:.1f}%\n"
                        f"AI: {synthetic * 100:.1f}%\n"
                        f"Threshold: {threshold * 100:.1f}%\n"
                        f"Streak: {self.alert_streak}/{CONSECUTIVE_ALERTS_REQUIRED}"
                    )
                    color = (0.7, 1, 0.8, 1) if synthetic < threshold else (1, 0.8, 0.3, 1)

                Clock.schedule_once(lambda _dt, t=text, c=color: self.update_analysis(t, c), 0)
            finally:
                if TEMP_FILE.exists():
                    TEMP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    VachaShieldApp().run()
