#!/usr/bin/env python3
"""
Praating-AI: Push-to-talk voice dictation tool.

Hold mouse button while speaking, release to transcribe.
Text is typed at the current cursor position.
"""
import os
import subprocess
import threading
import queue
import ctypes
from pathlib import Path

# Set up CUDA library paths BEFORE importing GPU libraries
def _setup_cuda_paths():
    """Configure LD_LIBRARY_PATH for CUDA libraries installed via pip."""
    try:
        import nvidia.cudnn
        import nvidia.cublas

        # Get library paths
        cudnn_path = getattr(nvidia.cudnn, '__path__', None)
        cublas_path = getattr(nvidia.cublas, '__path__', None)

        if cudnn_path:
            cudnn_path = os.path.join(cudnn_path[0], "lib")
        if cublas_path:
            cublas_path = os.path.join(cublas_path[0], "lib")

        # Load the libraries directly using ctypes
        for lib_path in [cudnn_path, cublas_path]:
            if lib_path and os.path.isdir(lib_path):
                for lib_file in sorted(os.listdir(lib_path)):
                    if lib_file.endswith('.so') or '.so.' in lib_file:
                        try:
                            ctypes.CDLL(os.path.join(lib_path, lib_file), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass  # Some libs may have missing deps, that's ok
    except ImportError:
        pass  # CUDA libs not installed, will use CPU

_setup_cuda_paths()

import yaml
import numpy as np
import sounddevice as sd
from pynput import mouse
from faster_whisper import WhisperModel

# Paths
PACKAGE_DIR = Path(__file__).parent
USER_CONFIG_DIR = Path.home() / ".config" / "praating-ai"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG_FILE = PACKAGE_DIR / "config.yaml"
SOUND_START = PACKAGE_DIR / "sounds" / "start.wav"
SOUND_STOP = PACKAGE_DIR / "sounds" / "stop.wav"

# Load configuration
def load_config():
    """Load configuration from YAML file.

    Checks ~/.config/praating-ai/config.yaml first, falls back to package default.
    """
    defaults = {
        "mode": "push_to_talk",
        "mouse_button": "button9",
        "model_size": "large-v3",
        "device": "cuda",
        "compute_type": "float32",
        "silence_threshold": 0.01,
        "silence_duration": 1.5,
        "add_trailing_space": True,
        "add_newline": False,
        "press_enter": False,
        "capitalize_first": True,
    }

    # Check user config first, then package default
    config_file = USER_CONFIG_FILE if USER_CONFIG_FILE.exists() else DEFAULT_CONFIG_FILE

    if config_file.exists():
        with open(config_file) as f:
            user_config = yaml.safe_load(f) or {}
            defaults.update(user_config)

    return defaults

config = load_config()

# Parse mouse button from config
def get_mouse_button(name):
    """Convert button name string to pynput Button."""
    button_map = {
        "left": mouse.Button.left,
        "right": mouse.Button.right,
        "middle": mouse.Button.middle,
    }
    if name in button_map:
        return button_map[name]
    # Handle button8, button9, etc.
    return getattr(mouse.Button, name, mouse.Button.button9)

MODE = config["mode"]
MOUSE_BUTTON = get_mouse_button(config["mouse_button"])
SAMPLE_RATE = 16000
MODEL_SIZE = config["model_size"]
DEVICE = config["device"]
COMPUTE_TYPE = config["compute_type"]

# Continuous mode settings
SILENCE_THRESHOLD = config["silence_threshold"]
SILENCE_DURATION = config["silence_duration"]

# Post-transcription behavior
ADD_TRAILING_SPACE = config["add_trailing_space"]
ADD_NEWLINE = config["add_newline"]
PRESS_ENTER = config["press_enter"]
CAPITALIZE_FIRST = config["capitalize_first"]


class Dictation:
    def __init__(self):
        print(f"Loading Whisper {MODEL_SIZE} model on {DEVICE}...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded. Ready for dictation.")

        if MODE == "continuous":
            print("Continuous mode: listening for speech...\n")
        else:
            print(f"Push-to-talk mode: hold mouse {MOUSE_BUTTON} while speaking.\n")

        self.recording = False
        self.audio_buffer = []
        self.stream = None

        # Continuous mode state
        self.is_speaking = False
        self.silence_samples = 0
        self.samples_per_second = SAMPLE_RATE

        # Queue for transcription results
        self.transcribe_queue = queue.Queue()

        # Start transcription worker thread
        self.worker = threading.Thread(target=self._transcribe_worker, daemon=True)
        self.worker.start()

    def _play_sound(self, sound_file):
        """Play a sound file using pw-play (PipeWire)."""
        threading.Thread(
            target=lambda: subprocess.run(
                ["pw-play", str(sound_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ),
            daemon=True
        ).start()

    def _audio_callback(self, indata, frames, time, status):
        """Called for each audio block while recording."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.audio_buffer.append(indata.copy())

    def _continuous_audio_callback(self, indata, frames, time, status):
        """Called for each audio block in continuous mode."""
        if status:
            print(f"Audio status: {status}")

        # Calculate audio level
        audio_level = np.abs(indata).mean()

        if audio_level > SILENCE_THRESHOLD:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self._play_sound(SOUND_START)
                print("Speech detected...")
            self.audio_buffer.append(indata.copy())
            self.silence_samples = 0
        elif self.is_speaking:
            # Still capturing, but silence
            self.audio_buffer.append(indata.copy())
            self.silence_samples += frames

            # Check if silence duration exceeded
            if self.silence_samples >= SILENCE_DURATION * self.samples_per_second:
                self._process_continuous_buffer()

    def _process_continuous_buffer(self):
        """Process accumulated audio in continuous mode."""
        self.is_speaking = False
        self.silence_samples = 0
        self._play_sound(SOUND_STOP)

        if self.audio_buffer:
            audio = np.concatenate(self.audio_buffer, axis=0).flatten()
            self.audio_buffer = []
            self.transcribe_queue.put(audio)
            print("Processing...")

    def start_recording(self):
        """Start recording audio."""
        if self.recording:
            return

        self.recording = True
        self.audio_buffer = []

        # Play start sound (non-blocking)
        self._play_sound(SOUND_START)

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self.stream.start()
        print("Recording...")

    def stop_recording(self):
        """Stop recording and queue for transcription."""
        if not self.recording:
            return

        self.recording = False

        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Play stop sound (non-blocking)
        self._play_sound(SOUND_STOP)

        # Get audio data
        if self.audio_buffer:
            audio = np.concatenate(self.audio_buffer, axis=0).flatten()
            self.transcribe_queue.put(audio)
            print("Processing...")
        else:
            print("No audio recorded.")

    def _transcribe_worker(self):
        """Worker thread that processes transcription queue."""
        while True:
            audio = self.transcribe_queue.get()
            if audio is None:
                break

            try:
                # Transcribe
                segments, info = self.model.transcribe(audio, beam_size=5)
                text = "".join(segment.text for segment in segments).strip()

                if text:
                    print(f"Transcribed: {text}")
                    self._type_text(text)
                else:
                    print("No speech detected.")
            except Exception as e:
                print(f"Transcription error: {e}")

    def _type_text(self, text):
        """Type text at current cursor position using xdotool."""
        try:
            # Apply text transformations
            if text:
                if CAPITALIZE_FIRST:
                    text = text[0].upper() + text[1:]
                else:
                    text = text[0].lower() + text[1:]
            if ADD_TRAILING_SPACE:
                text += " "
            if ADD_NEWLINE:
                text += "\n"

            # Use xdotool to type the text
            # --clearmodifiers ensures hotkey modifiers don't interfere
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", text],
                check=True
            )

            # Press Enter if configured
            if PRESS_ENTER:
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "Return"],
                    check=True
                )
        except subprocess.CalledProcessError as e:
            print(f"xdotool error: {e}")
        except FileNotFoundError:
            print("Error: xdotool not found. Install with: sudo apt install xdotool")

    def run(self):
        """Start listening based on configured mode."""
        if MODE == "continuous":
            self._run_continuous()
        else:
            self._run_push_to_talk()

    def _run_push_to_talk(self):
        """Push-to-talk mode: hold mouse button to record."""
        def on_click(x, y, button, pressed):
            if button == MOUSE_BUTTON:
                if pressed:
                    self.start_recording()
                else:
                    self.stop_recording()

        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

    def _run_continuous(self):
        """Continuous mode: always listening, detect speech by volume."""
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_samples = 0

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self._continuous_audio_callback
        )

        with stream:
            print("Listening... (Ctrl+C to stop)")
            while True:
                sd.sleep(100)


def main():
    try:
        dictation = Dictation()
        dictation.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
