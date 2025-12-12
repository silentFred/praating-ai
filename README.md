# Praating-AI

Push-to-talk voice dictation for Linux. Hold a button, speak, release — text appears at your cursor.

Built for use with Claude Code, terminals, text editors, and other text application.

## Quick Start

### 1. Install system dependencies

```bash
sudo apt install xdotool portaudio19-dev libasound2-dev
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
./run.sh
```

First run downloads the Whisper model (~1.5GB). Subsequent starts are fast.

### 4. Use it

**Hold your mouse button (default: button9) while speaking, release to transcribe.**

Text is typed wherever your cursor is focused.

## Configuration

Edit `config.yaml` to customize:

```yaml
# Input settings
mode: push_to_talk          # push_to_talk or continuous
mouse_button: button9       # button8, button9, left, right, middle (push_to_talk mode)
model_size: medium          # tiny, base, small, medium, large-v3
device: cuda                # cuda (GPU) or cpu
compute_type: float32       # float32, float16, int8

# Continuous mode settings (always listening, auto-detects speech)
silence_threshold: 0.01     # Audio level below this is silence (0.0-1.0)
silence_duration: 1.5       # Seconds of silence before processing

# Output behavior
add_trailing_space: true    # Add a space after transcribed text
add_newline: false          # Add a newline after transcribed text
press_enter: false          # Press Enter after transcribing (useful for chat apps)
capitalize_first: true      # Capitalize first letter of transcription
```

### Modes

- **push_to_talk**: Hold mouse button while speaking, release to transcribe
- **continuous**: Always listening, automatically detects speech and transcribes when you pause

### Finding your mouse button

Run this to detect which button number your mouse uses:

```bash
python3 -c "from pynput import mouse
print('Click your button...')
with mouse.Listener(on_click=lambda x,y,b,p: print(b) or not p) as l: l.join()"
```

## Requirements

- Linux with X11 (uses xdotool for typing)
- Python 3.8+
- Microphone
- NVIDIA GPU recommended (falls back to CPU)

## How It Works

1. Loads Whisper speech recognition model on startup
2. Listens for mouse button press → starts recording (beep)
3. On button release → stops recording (beep)
4. Transcribes audio using faster-whisper
5. Types the text at cursor position using xdotool

## Troubleshooting

### No GPU acceleration
Make sure CUDA libraries are loaded via `run.sh`. Running `python dictate.py` directly may not find them.

### Text not appearing
- Ensure xdotool is installed
- Check you're using X11 (not Wayland): `echo $XDG_SESSION_TYPE`
- For Wayland, you'd need to use `ydotool` or `wtype` instead

### Wrong mouse button
Use the detection snippet above to find your button, then update `MOUSE_BUTTON` in `dictate.py`.
