# Praating-AI

Push-to-talk voice dictation for Linux. Hold a button, speak, release — text appears at your cursor.

Built for use with Claude Code, terminals, text editors, and any application.

## Installation

### 1. Install system dependencies

```bash
sudo apt install xdotool portaudio19-dev libasound2-dev
```

### 2. Install praating-ai

**With GPU support (recommended):**
```bash
pip install praating-ai[cuda]
```

**CPU only:**
```bash
pip install praating-ai
```

Or install from GitHub:
```bash
pip install git+https://github.com/silentFred/praating-ai.git
```

### 3. Run

```bash
praating
```

First run downloads the Whisper model (~3GB for large-v3). Subsequent starts are fast.

> **Note:** For GPU acceleration, use the wrapper script which sets up CUDA library paths:
> ```bash
> ./run.sh
> ```

## Usage

**Hold your mouse button (default: button9) while speaking, release to transcribe.**

Text is typed wherever your cursor is focused.

## Configuration

Create `~/.config/praating-ai/config.yaml` to customize:

```yaml
# Input settings
mode: push_to_talk          # push_to_talk or continuous
mouse_button: button9       # button8, button9, left, right, middle
model_size: large-v3        # tiny, base, small, medium, large-v3
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
Make sure CUDA libraries are loaded. Use `./run.sh` or set `LD_LIBRARY_PATH` to include the nvidia cudnn/cublas lib directories.

### Text not appearing
- Ensure xdotool is installed: `sudo apt install xdotool`
- Check you're using X11: `echo $XDG_SESSION_TYPE`
- For Wayland, you'd need to use `ydotool` or `wtype` instead

### Wrong mouse button
Use the detection snippet above to find your button, then update `mouse_button` in your config.
