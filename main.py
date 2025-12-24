import numpy as np
import nengo

from audio_input import audio_q, start_stream, SAMPLE_RATE, BLOCK_SIZE

# Nengo's simulation time step.
# We choose dt so that each Nengo step corresponds to exactly one audio block.
DT = BLOCK_SIZE / SAMPLE_RATE  # ~0.016 s (16 ms)

# ---------------------------------------------------------------------------
# Calibration parameters
# ---------------------------------------------------------------------------

# We assume:
# - channel 0 is the Mac microphone
# - channel 1 is the iPhone microphone
#
# The two mics do not have exactly the same sensitivity or processing,
# so we scale them to make them more comparable.

# Scale factor for the Mac microphone signal (channel 0).
# A value < 1.0 means we reduce its volume (to make it less dominant).
MAC_GAIN = 0.3

# Scale factor for the iPhone microphone signal (channel 1).
# A value > 1.0 means we boost its volume (to compensate for noise cancelling
# or lower sensitivity).
IPHONE_GAIN = 3.0

# After gain calibration there may still be a constant bias in ILD.
# ILD_OFFSET shifts all ILD values by a fixed amount (in dB).
# This is used so that speaking in the middle between the mics
# gives an angle around 0 degrees.
ILD_OFFSET = 25.0  # dB

# If the total energy (both channels together) is very small, we treat it as
# "near silence". In that case we do NOT update the estimate, to avoid the
# angle jumping around when there is almost no sound.
ENERGY_THRESHOLD = 1e-4

# ---------------------------------------------------------------------------
# Live audio feature extraction: ILD (Interaural Level Difference)
# ---------------------------------------------------------------------------

# This variable stores the last valid ILD value.
# When there is no new data (or the sound is too quiet), we keep this value.
last_feat = np.array([0.0])


def feature_node_func(t):
    """
    Nengo Node function that runs once per simulation step.

    It tries to:
    1. Read one audio block from the queue.
    2. Separate the two channels (Mac and iPhone).
    3. Apply gain calibration to each channel.
    4. Compute the average energy in each channel.
    5. Compute the ILD (in dB) = 10 * log10(energy_left / energy_right).
    6. Apply an offset to ILD so that "center" ≈ 0.
    7. Return ILD as a 1D array.

    If there is no new audio block or the sound is too quiet,
    it returns the last valid ILD (keeps the estimate).
    """
    global last_feat

    try:
        # Try to get one audio block from the queue without waiting.
        # If the queue is empty, this raises queue.Empty and jumps to except.
        block = audio_q.get_nowait()  # shape: (BLOCK_SIZE, 2)

        # Split into channels:
        # channel 0 = Mac, channel 1 = iPhone
        mac = block[:, 0]
        iphone = block[:, 1]

        # Apply channel-specific gains to compensate for different sensitivity.
        mac = MAC_GAIN * mac
        iphone = IPHONE_GAIN * iphone

        # Compute mean squared amplitude (energy) for each channel.
        # Add a tiny epsilon so we never divide by zero.
        eps = 1e-8
        L = np.mean(mac**2) + eps     # energy of Mac channel
        R = np.mean(iphone**2) + eps  # energy of iPhone channel
        total_energy = L + R

        # If almost no energy (near silence), keep previous ILD.
        if total_energy < ENERGY_THRESHOLD:
            return last_feat

        # ILD in decibels: positive means Mac louder, negative means iPhone louder
        ild = 10 * np.log10(L / R)

        # Apply the calibrated offset to shift the "center" towards 0.
        ild = ild + ILD_OFFSET

        # Store this as the latest valid feature.
        last_feat = np.array([ild])

    except Exception:
        # If no new audio block is available, or something went wrong,
        # keep using the last previous ILD estimate.
        pass

    # Return a 1D numpy array, as expected by Nengo Node (size_out = 1).
    return last_feat


def ild_to_angle(ild):
    """
    Convert ILD (in dB) to a rough angle estimate in degrees.

    - We clip ILD to the range [-40, 40] dB to avoid huge outliers.
    - We then linearly map:
        -40 dB -> -90 degrees (e.g. far to one side)
        +40 dB -> +90 degrees (far to the other side)

    This mapping is not physically perfect, but it's simple and works
    as a demonstration of how ILD can be turned into a direction estimate.
    """
    ild_clipped = max(-40.0, min(40.0, float(ild)))
    angle_deg = (ild_clipped / 40.0) * 90.0
    return angle_deg


# ---------------------------------------------------------------------------
# Nengo model definition
# ---------------------------------------------------------------------------

# Create a new Nengo network. This is the "container" for all nodes and connections.
model = nengo.Network(label="Live ILD localization")

with model:
    # Node that produces the ILD feature (in dB) each time step.
    # Nengo will call feature_node_func(t) every dt seconds.
    feat = nengo.Node(feature_node_func, size_out=1, label="ILD_dB")

    # Node that receives the angle estimate (in degrees).
    # It has no output function; it is just a sink for the mapping below.
    angle = nengo.Node(size_in=1, label="angle_deg")

    # Connect feature node to angle node.
    # The 'function' argument tells Nengo how to convert ILD to angle.
    # Nengo internally computes a set of decoders so that the node
    # approximately computes ild_to_angle(x[0]).
    nengo.Connection(
        feat,
        angle,
        function=lambda x: ild_to_angle(x[0])
    )

    # Optional probes so we can record data for offline analysis.
    # p_feat records the ILD over time.
    # p_angle records the angle (smoothed by synapse=0.1 s).
    p_feat = nengo.Probe(feat, synapse=None)
    p_angle = nengo.Probe(angle, synapse=0.1)

# ---------------------------------------------------------------------------
# Start the audio stream in the same process as Nengo
# ---------------------------------------------------------------------------

# We start the live audio stream immediately when main.py is imported.
# This ensures that the audio queue is filled while the Nengo model is running,
# whether we run this file directly or via Nengo GUI.
stream = start_stream(device=5, channels=2)
print(f"Stream started (SR={SAMPLE_RATE}, DT={DT:.4f}s)")

# ---------------------------------------------------------------------------
# Optional: standalone test without Nengo GUI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run a 5-second simulation using the same dt as our audio blocks.
    with nengo.Simulator(model, dt=DT) as sim:
        sim.run(5.0)

    # Stop and close the audio stream when we're done.
    stream.stop()
    stream.close()

    # Print the first 20 recorded angle values to the terminal.
    print(sim.data[p_angle][:20])
