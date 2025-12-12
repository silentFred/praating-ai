#!/usr/bin/env python3
"""GUI setup wizard for Praating-AI."""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import yaml
from pynput import mouse

# Config paths
USER_CONFIG_DIR = Path.home() / ".config" / "praating-ai"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"


def load_config():
    """Load existing config or return defaults."""
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

    if USER_CONFIG_FILE.exists():
        with open(USER_CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}
            defaults.update(user_config)

    return defaults


def save_config(settings):
    """Save settings to user config file."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(USER_CONFIG_FILE, 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)


class SetupGUI:
    def __init__(self):
        self.config = load_config()
        self.detected_button = None
        self.mouse_listener = None

        # Create main window
        self.root = tk.Tk()
        self.root.title("Praating-AI Setup")
        self.root.resizable(False, False)

        # Center window
        self.root.eval('tk::PlaceWindow . center')

        self._create_widgets()

    def _create_widgets(self):
        # Main frame with padding
        main = ttk.Frame(self.root, padding=20)
        main.grid(row=0, column=0, sticky="nsew")

        row = 0

        # Title
        title = ttk.Label(main, text="Praating-AI Setup", font=("", 14, "bold"))
        title.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Mouse button section
        ttk.Label(main, text="Push-to-talk button:").grid(row=row, column=0, sticky="w", pady=5)

        self.button_var = tk.StringVar(value=self.config["mouse_button"])
        self.button_entry = ttk.Entry(main, textvariable=self.button_var, width=15)
        self.button_entry.grid(row=row, column=1, pady=5, padx=5)

        detect_btn = ttk.Button(main, text="Detect", command=self._start_detection)
        detect_btn.grid(row=row, column=2, pady=5)
        row += 1

        # Detection status
        self.detect_label = ttk.Label(main, text="", foreground="gray")
        self.detect_label.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        # Separator
        ttk.Separator(main, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Model size
        ttk.Label(main, text="Model size:").grid(row=row, column=0, sticky="w", pady=5)
        self.model_var = tk.StringVar(value=self.config["model_size"])
        model_combo = ttk.Combobox(main, textvariable=self.model_var, width=12, state="readonly")
        model_combo["values"] = ("tiny", "base", "small", "medium", "large-v3")
        model_combo.grid(row=row, column=1, columnspan=2, sticky="w", pady=5)
        row += 1

        # Device
        ttk.Label(main, text="Device:").grid(row=row, column=0, sticky="w", pady=5)
        self.device_var = tk.StringVar(value=self.config["device"])
        device_combo = ttk.Combobox(main, textvariable=self.device_var, width=12, state="readonly")
        device_combo["values"] = ("cuda", "cpu")
        device_combo.grid(row=row, column=1, columnspan=2, sticky="w", pady=5)
        row += 1

        # Mode
        ttk.Label(main, text="Mode:").grid(row=row, column=0, sticky="w", pady=5)
        self.mode_var = tk.StringVar(value=self.config["mode"])
        mode_combo = ttk.Combobox(main, textvariable=self.mode_var, width=12, state="readonly")
        mode_combo["values"] = ("push_to_talk", "continuous")
        mode_combo.grid(row=row, column=1, columnspan=2, sticky="w", pady=5)
        row += 1

        # Separator
        ttk.Separator(main, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Checkboxes
        self.space_var = tk.BooleanVar(value=self.config["add_trailing_space"])
        ttk.Checkbutton(main, text="Add trailing space", variable=self.space_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2
        )
        row += 1

        self.enter_var = tk.BooleanVar(value=self.config["press_enter"])
        ttk.Checkbutton(main, text="Press Enter after transcribing", variable=self.enter_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2
        )
        row += 1

        self.caps_var = tk.BooleanVar(value=self.config["capitalize_first"])
        ttk.Checkbutton(main, text="Capitalize first letter", variable=self.caps_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2
        )
        row += 1

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(20, 0))

        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.root.quit).pack(side="left", padx=5)

    def _start_detection(self):
        """Start listening for mouse button click."""
        self.detect_label.config(text="Click your push-to-talk button...", foreground="blue")
        self.root.update()

        def on_click(x, y, button, pressed):
            if pressed and button not in (mouse.Button.left, mouse.Button.right):
                self.detected_button = button
                return False  # Stop listener

        self.mouse_listener = mouse.Listener(on_click=on_click)
        self.mouse_listener.start()

        # Poll for result
        self._check_detection()

    def _check_detection(self):
        """Check if button was detected."""
        if self.mouse_listener and not self.mouse_listener.is_alive():
            if self.detected_button:
                button_name = self.detected_button.name
                self.button_var.set(button_name)
                self.detect_label.config(text=f"Detected: {button_name}", foreground="green")
            self.mouse_listener = None
        elif self.mouse_listener:
            self.root.after(100, self._check_detection)

    def _save(self):
        """Save configuration and close."""
        self.config.update({
            "mouse_button": self.button_var.get(),
            "model_size": self.model_var.get(),
            "device": self.device_var.get(),
            "mode": self.mode_var.get(),
            "add_trailing_space": self.space_var.get(),
            "press_enter": self.enter_var.get(),
            "capitalize_first": self.caps_var.get(),
        })

        save_config(self.config)
        messagebox.showinfo("Saved", f"Config saved to:\n{USER_CONFIG_FILE}\n\nRun 'praating' to start!")
        self.root.quit()

    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def main():
    """Launch the setup GUI."""
    app = SetupGUI()
    app.run()


if __name__ == "__main__":
    main()
