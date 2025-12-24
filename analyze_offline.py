# ============================================================================
# OFFLINE ANALYSIS: Record audio and plot direction estimation
# ============================================================================
# This script records live audio for a fixed duration, runs the Nengo SNN
# model offline, and plots the results as waveforms + direction over time.

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from audio_input import SAMPLE_RATE, BLOCK_SIZE
from main import (ild_to_angle, MAC_GAIN, IPHONE_GAIN, ILD_OFFSET, ENERGY_THRESHOLD, DEVICE_INDEX)

# ============================================================================
# CONFIGURATION
# ============================================================================

DURATION = 10.0  # Duration to record in seconds
CHANNELS = 2  # 2 input channels (Mac + iPhone)

# Labels for your devices
LEFT_DEVICE = "iPhone"  # Channel 0
RIGHT_DEVICE = "MacBook"  # Channel 1

# Threshold in dB for direction classification (optional, for step plot)
THRESHOLD_DB = 3.0

# ============================================================================
# RECORD AUDIO
# ============================================================================

print(f"Recording {DURATION} seconds of audio from device {DEVICE_INDEX}...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=CHANNELS, device=DEVICE_INDEX)
sd.wait()
print("Recording complete.")

# Time axis for raw waveform
t_waveform = np.arange(audio.shape[0]) / SAMPLE_RATE

# ============================================================================
# COMPUTE ILD AND ANGLE OFFLINE
# ============================================================================

# We'll process the recorded audio in blocks (just like in live mode)
# and compute the angle for each block.

block_times = []
ild_values = []
angle_values = []
direction_values = []

n_blocks = audio.shape[0] // BLOCK_SIZE

for i in range(n_blocks):
    start_idx = i * BLOCK_SIZE
    end_idx = start_idx + BLOCK_SIZE

    block = audio[start_idx:end_idx, :]

    # Manually compute ILD (same as feature_node_func)
    mac = block[:, 0]
    iphone = block[:, 1]

    # Apply gains
    mac = MAC_GAIN * mac
    iphone = IPHONE_GAIN * iphone

    eps = 1e-8
    L = np.mean(mac ** 2) + eps
    R = np.mean(iphone ** 2) + eps
    total_energy = L + R

    # Skip if silent
    if total_energy < ENERGY_THRESHOLD:
        continue

    ild = 10 * np.log10(L / R)
    ild = ild + ILD_OFFSET

    angle = ild_to_angle(ild)

    # Determine direction based on threshold
    if ild > THRESHOLD_DB:
        direction = -1  # Left (iPhone louder)
    elif ild < -THRESHOLD_DB:
        direction = +1  # Right (Mac louder)
    else:
        direction = 0  # Center

    # Store results
    time_block = (start_idx + BLOCK_SIZE / 2) / SAMPLE_RATE
    block_times.append(time_block)
    ild_values.append(ild)
    angle_values.append(angle)
    direction_values.append(direction)

# Convert to numpy arrays
block_times = np.array(block_times)
ild_values = np.array(ild_values)
angle_values = np.array(angle_values)
direction_values = np.array(direction_values)

# ============================================================================
# PLOTTING
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

# Plot 1: Waveforms from both devices
ax1.plot(t_waveform, audio[:, 0], label=f"{LEFT_DEVICE} (left)", color='blue', alpha=0.7)
ax1.plot(t_waveform, audio[:, 1], label=f"{RIGHT_DEVICE} (right)", color='orange', alpha=0.7)
ax1.set_ylabel("Amplitude")
ax1.set_title(f"Waveform – {LEFT_DEVICE} vs {RIGHT_DEVICE}")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Estimated direction over time
ax2.step(block_times, direction_values, where="mid", linewidth=2, color='steelblue')
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["Left", "Center", "Right"])
ax2.set_ylabel("Direction")
ax2.set_xlabel("Time [s]")
ax2.set_title(
    f"Estimated direction (Left = {LEFT_DEVICE}, Right = {RIGHT_DEVICE}, "
    f"threshold = {THRESHOLD_DB} dB)"
)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print(f"\nSummary Statistics:")
print(f"  Total duration: {DURATION} s")
print(f"  Total blocks processed: {len(block_times)}")
print(f"  Mean ILD: {np.mean(ild_values):.2f} dB")
print(f"  ILD std: {np.std(ild_values):.2f} dB")
print(f"  Mean angle: {np.mean(angle_values):.2f}°")
print(f"  Direction breakdown:")
print(f"    - Left ({LEFT_DEVICE}): {np.sum(direction_values == -1)} frames")
print(f"    - Center: {np.sum(direction_values == 0)} frames")
print(f"    - Right ({RIGHT_DEVICE}): {np.sum(direction_values == 1)} frames")
