# ============================================================================
# LIVE DETECTION: Real-time sound localization with PyQtGraph GUI
# ============================================================================
# This script records live audio from the Aggregate Device (iPhone + MacBook)
# and displays the waveforms and direction estimation in real-time.

import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore

# Import your configuration from main.py
from main import (SAMPLE_RATE, BLOCK_SIZE, DEVICE_INDEX)

# ---------- LIVE DETECTION CONFIG ----------
BUFFER_SECONDS = 5.0  # rolling window for waveform display
WINDOW_MS = 50  # window size for ILD computation
HOP_MS = 20  # hop size between windows
THRESHOLD_DB = 3.0
# ------------------------------------------

# ---------- DEVICE MAPPING ----------
# Channel 0 = iPhone Microphone
# Channel 1 = MacBook Pro Microphone
LEFT_DEVICE = "iphone"
RIGHT_DEVICE = "macbook"

device_to_channel = {
    "iphone": 0,
    "macbook": 1,
}

left_ch = device_to_channel[LEFT_DEVICE.lower()]
right_ch = device_to_channel[RIGHT_DEVICE.lower()]

left_label = f"{LEFT_DEVICE.capitalize()} (left)"
right_label = f"{RIGHT_DEVICE.capitalize()} (right)"
# ------------------------------------

# ---------- AUDIO BUFFER ----------
buffer_samples = int(BUFFER_SECONDS * SAMPLE_RATE)
audio_buffer = np.zeros((buffer_samples, 2), dtype=np.float32)


def audio_callback(indata, frames, time, status):
    """Rolling buffer updated on every audio chunk."""
    global audio_buffer
    if status:
        print("Audio status:", status)

    frames = min(frames, buffer_samples)
    audio_buffer = np.roll(audio_buffer, -frames, axis=0)
    audio_buffer[-frames:, :] = indata[-frames:, :]


# ---------- SOUNDDEVICE INIT ----------
sd.default.device = DEVICE_INDEX
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 2

stream = sd.InputStream(
    callback=audio_callback,
    blocksize=BLOCK_SIZE,
)
stream.start()

print(f"Recording from device {DEVICE_INDEX} ({LEFT_DEVICE} + {RIGHT_DEVICE})...")
print(f"Buffer: {BUFFER_SECONDS} seconds | Threshold: {THRESHOLD_DB} dB\n")

# ---------- PYQTGRAPH UI ----------
app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(
    show=True,
    title=f"Live Sound Localization ({LEFT_DEVICE.capitalize()} vs {RIGHT_DEVICE.capitalize()})"
)
win.resize(1200, 700)

# ===== Plot 1: Waveforms =====
p_wave = win.addPlot(title=f"Waveforms – {left_label} vs {right_label}")
p_wave.showGrid(x=True, y=True)
p_wave.setLabel('bottom', 'Time', units='s')
p_wave.setLabel('left', 'Amplitude')
wave_left = p_wave.plot(pen=pg.mkPen('blue', width=2))
wave_right = p_wave.plot(pen=pg.mkPen('orange', width=2))

# ===== Plot 2: Direction Indicator =====
win.nextRow()
p_dir = win.addPlot(title=f"Estimated Direction (threshold = {THRESHOLD_DB} dB)")
p_dir.setYRange(-1.5, 1.5)
p_dir.setXRange(-BUFFER_SECONDS, 0)
p_dir.showGrid(x=True, y=True)
p_dir.setLabel('bottom', 'Time', units='s')
p_dir.setLabel('left', 'Direction')
direction_curve = p_dir.plot(pen=pg.mkPen('steelblue', width=2.5))

# ===== Plot 3: ILD (Interaural Level Difference) =====
win.nextRow()
p_ild = win.addPlot(title="ILD – Interaural Level Difference (dB)")
p_ild.setXRange(-BUFFER_SECONDS, 0)
p_ild.enableAutoRange(axis='y')  # Enable auto-range for Y axis
p_ild.showGrid(x=True, y=True)
p_ild.setLabel('bottom', 'Time', units='s')
p_ild.setLabel('left', 'ILD', units='dB')
p_ild.addLine(y=THRESHOLD_DB, pen=pg.mkPen('red', width=1, style=QtCore.Qt.DashLine))
p_ild.addLine(y=-THRESHOLD_DB, pen=pg.mkPen('red', width=1, style=QtCore.Qt.DashLine))
ild_curve = p_ild.plot(pen=pg.mkPen('green', width=2))

# ===== Text Overlay (Direction Label) =====
text_item = pg.TextItem("", color=(255, 255, 255), anchor=(0.5, 0.5))
text_item.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
p_dir.addItem(text_item)
text_item.setPos(0, 0)

# ===== Statistics Text =====
stats_text = pg.TextItem("", color=(200, 200, 200), anchor=(1, 0))
stats_text.setFont(QtGui.QFont("Courier", 9))
p_ild.addItem(stats_text)
stats_text.setPos(-0.1, np.inf)


def update():
    """Update all plots with current audio buffer data."""
    global audio_buffer

    buf = audio_buffer.copy()

    # Time axis for waveforms (in seconds, from -BUFFER_SECONDS to 0)
    t_wave = np.linspace(-BUFFER_SECONDS, 0, buffer_samples)

    # Update waveform plots
    wave_left.setData(t_wave, buf[:, left_ch])
    wave_right.setData(t_wave, buf[:, right_ch])

    # ---- Compute direction and ILD ----
    win_len = int(WINDOW_MS / 1000.0 * SAMPLE_RATE)
    hop_len = int(HOP_MS / 1000.0 * SAMPLE_RATE)

    indices = np.arange(0, buf.shape[0] - win_len, hop_len)
    times = -BUFFER_SECONDS + (indices + win_len / 2) / SAMPLE_RATE

    directions = []
    ilds = []
    eps = 1e-12

    for start in indices:
        frame = buf[start:start + win_len]

        # Get energy of both channels
        left_sig = frame[:, left_ch]
        right_sig = frame[:, right_ch]

        rms_left = np.sqrt(np.mean(left_sig ** 2) + eps)
        rms_right = np.sqrt(np.mean(right_sig ** 2) + eps)

        # Compute ILD
        ild = 20 * np.log10(rms_left / rms_right)
        ilds.append(ild)

        # Classify direction based on threshold
        if ild > THRESHOLD_DB:
            directions.append(-1)  # left louder
        elif ild < -THRESHOLD_DB:
            directions.append(1)  # right louder
        else:
            directions.append(0)  # center

    if len(times) > 0:
        # Update direction curve
        direction_curve.setData(times, directions)

        # Update ILD curve
        ild_curve.setData(times, ilds)

        # Dynamically adjust ILD Y-range
        if len(ilds) > 0:
            ild_min = np.min(ilds)
            ild_max = np.max(ilds)
            ild_range = ild_max - ild_min
            margin = ild_range * 0.1 if ild_range > 0 else 5
            p_ild.setYRange(ild_min - margin, ild_max + margin)

        # Update text label (most recent direction)
        last_direction = directions[-1]
        if last_direction == -1:
            text_item.setText("← LEFT")
            text_item.setColor((70, 150, 255))
        elif last_direction == 1:
            text_item.setText("RIGHT →")
            text_item.setColor((255, 150, 70))
        else:
            text_item.setText("CENTER")
            text_item.setColor((100, 200, 100))

        # Update statistics
        mean_ild = np.mean(ilds)
        std_ild = np.std(ilds)
        last_ild = ilds[-1]

        stats_text.setText(
            f"Mean ILD: {mean_ild:+.2f} dB\n"
            f"Std ILD:  {std_ild:.2f} dB\n"
            f"Last ILD: {last_ild:+.2f} dB"
        )


# ---------- TIMER FOR UPDATES ----------
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)  # ~30 FPS

# ---------- RUN APP ----------
try:
    QtWidgets.QApplication.instance().exec()
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    stream.stop()
    stream.close()
    print("Stream closed. Done!")
