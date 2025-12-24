# ============================================================================
# NEUROMORPHIC SOUND LOCALIZATION USING SPIKING NEURAL NETWORKS
# ============================================================================
# This program uses spiking neurons (like those in the brain's auditory system)
# to estimate the direction of a sound source using two microphones.

import numpy as np
import nengo

# Import audio and timing constants from our audio module
from audio_input import audio_q, start_stream, SAMPLE_RATE, BLOCK_SIZE

# ============================================================================
# FUNDAMENTAL TIMING PARAMETER
# ============================================================================

# Nengo's simulation timestep: how often we update the neural network.
# We set it to match one audio block duration so the neurons process
# fresh audio data every simulation step.
DT = BLOCK_SIZE / SAMPLE_RATE  # ~0.016 seconds (16 milliseconds)

# ============================================================================
# CALIBRATION PARAMETERS
# ============================================================================
# These values compensate for differences between the Mac and iPhone mics.
# Tune them so that speaking in the middle between the devices gives ~0 degrees.

# Scale factor for Mac microphone (channel 0).
# A value less than 1.0 reduces the Mac's sensitivity.
MAC_GAIN = 0.3

# Scale factor for iPhone microphone (channel 1).
# A value greater than 1.0 amplifies the iPhone's sensitivity.
IPHONE_GAIN = 3.0

# A fixed offset added to the ILD (in decibels) to shift the "center" point.
# If centered speech gives -10 degrees, increase this value.
ILD_OFFSET = 25.0

# Energy threshold below which we ignore audio (near silence).
# Avoids angle estimates jumping around when there is almost no sound.
ENERGY_THRESHOLD = 1e-5

# Store the last valid ILD value so we can return it if no new audio arrives.
last_feat = np.array([0.0])

DEVICE_INDEX = 5

# ============================================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================================


def feature_node_func(t):
    """
    Extract ILD (Interaural Level Difference) from live audio.

    This function runs once per simulation timestep (every ~16 ms).
    It reads the latest audio block from the queue and computes how much
    louder the sound is on the Mac side vs. the iPhone side.

    Parameters
    ----------
    t : float
        Current simulation time (not used here, but required by Nengo).

    Returns
    -------
    np.ndarray
        A 1D array containing the ILD value in decibels.
    """
    global last_feat  # Use the module-level variable to store the last value

    try:
        # Try to read one audio block (256 samples, 2 channels) from the queue.
        # If no block is available, this raises an exception and jumps to except.
        block = audio_q.get_nowait()  # shape: (BLOCK_SIZE, 2)

        # Separate the two channels: Mac and iPhone.
        mac = block[:, 0]
        iphone = block[:, 1]

        # Apply per-channel gain to make the microphones more similar in sensitivity.
        mac = MAC_GAIN * mac
        iphone = IPHONE_GAIN * iphone

        # Compute the average power (energy) in each channel.
        # Power = mean of squared amplitude.
        eps = 1e-8  # Tiny value to avoid dividing by zero
        L = np.mean(mac ** 2) + eps  # Energy of Mac channel
        R = np.mean(iphone ** 2) + eps  # Energy of iPhone channel
        total_energy = L + R  # Combined energy

        # If the sound is very quiet (near silence), keep the previous ILD estimate
        # instead of computing a noisy one. This keeps the angle stable.
        if total_energy < ENERGY_THRESHOLD:
            return last_feat

        # ILD in decibels: how many dB louder one side is than the other.
        # Positive ILD = Mac louder, Negative ILD = iPhone louder.
        ild = 10 * np.log10(L / R)

        # Apply the calibration offset so that "centered" audio gives ~0 ILD.
        ild = ild + ILD_OFFSET

        # Store this as the latest valid feature.
        last_feat = np.array([ild])

    except Exception:
        # If no new audio block is available, or an error occurred,
        # just keep using the last valid ILD estimate. This smooths the output.
        pass

    # Return the ILD as a 1D numpy array (required by Nengo Node with size_out=1).
    return last_feat


def ild_to_angle(ild):
    """
    Convert ILD (in decibels) to an angle estimate (in degrees).

    This is a simple linear mapping:
    - ILD = -40 dB (iPhone very loud) → angle = -90° (far left)
    - ILD = 0 dB (equal loudness) → angle = 0° (center)
    - ILD = +40 dB (Mac very loud) → angle = +90° (far right)

    Parameters
    ----------
    ild : float
        Interaural level difference in decibels.

    Returns
    -------
    float
        Estimated angle in degrees, range [-90, +90].
    """
    # Clip ILD to a reasonable range to avoid extreme outliers.
    ild_clipped = max(-40.0, min(40.0, float(ild)))

    # Linear map from [-40, 40] dB to [-90, 90] degrees.
    angle_deg = (ild_clipped / 40.0) * 90.0

    return angle_deg


# ============================================================================
# NENGO MODEL DEFINITION: SPIKING NEURAL NETWORK
# ============================================================================

# Create the main neural network container.
# All nodes, ensembles, and connections will be added inside the "with" block.
model = nengo.Network(label="Neuromorphic Sound Localization (SNN)")

with model:
    # ========================================================================
    # LAYER 0: INPUT STAGE
    # ========================================================================
    # Convert raw audio into a feature (ILD) that the brain-like network can use.

    # This node produces one ILD value per simulation timestep.
    # It's not spiking; it's just a signal source that feeds the SNN.
    feat_node = nengo.Node(
        feature_node_func,
        size_out=1,
        label="ILD_input_dB"
    )

    # ========================================================================
    # LAYER 1: PRIMARY SPIKING ENCODING
    # ========================================================================
    # The first spiking layer encodes the ILD feature into spike patterns.
    # Think of this as the auditory nerve or inferior colliculus receiving
    # the interaural difference signal.

    ild_snn = nengo.Ensemble(
        n_neurons=200,  # Use 200 spiking neurons to represent the ILD
        dimensions=1,  # The ILD is one-dimensional
        label="ILD_SpikeLayer",
        neuron_type=nengo.LIF(),  # Use Leaky Integrate-and-Fire (spiking) neurons
        # LIF neurons fire discrete spikes when their membrane potential crosses a threshold.
        # This is closer to biological neurons than rate-coded neurons.
        max_rates=nengo.dists.Uniform(100, 200),  # Each neuron fires 100-200 spikes/sec
    )

    # Connect the ILD input to the spiking layer.
    # synapse=0.01 means we use a 10 ms exponential decay filter for smoothing.
    nengo.Connection(feat_node, ild_snn, synapse=0.01)

    # ========================================================================
    # LAYER 2: ON/OFF NEURON PAIRS (from your lecture slides)
    # ========================================================================
    # The auditory system uses separate "On" and "Off" neuron populations:
    # - On neurons fire when the signal increases (positive changes in ILD)
    # - Off neurons fire when the signal decreases (negative changes in ILD)
    # Together, they provide redundancy and robust encoding of the signal.

    # ON-neurons: respond primarily to increases in sound level on the Mac side.
    on_neurons = nengo.Ensemble(
        n_neurons=100,
        dimensions=1,
        label="On_Neurons",
        neuron_type=nengo.LIF(),  # Also spiking
        max_rates=nengo.dists.Uniform(100, 200),
    )

    # OFF-neurons: respond primarily to decreases in sound level on the Mac side
    # (i.e., relative increases on the iPhone side).
    off_neurons = nengo.Ensemble(
        n_neurons=100,
        dimensions=1,
        label="Off_Neurons",
        neuron_type=nengo.LIF(),  # Also spiking
        max_rates=nengo.dists.Uniform(100, 200),
    )

    # Connect the ILD encoding layer to both on and off neurons.
    # Both populations receive the same input but will respond differently:
    # On neurons will fire more for positive ILD, off neurons for negative ILD.
    nengo.Connection(ild_snn, on_neurons, synapse=0.01)
    nengo.Connection(ild_snn, off_neurons, synapse=0.01)

    # ========================================================================
    # LAYER 3: ANGLE DECODING FROM SPIKE PATTERNS
    # ========================================================================
    # This ensemble decodes the spike patterns from the on/off neurons
    # to estimate the angle.
    # Nengo automatically learns a linear decoder that maps spike patterns to angles.

    angle_snn = nengo.Ensemble(
        n_neurons=150,  # 150 neurons to decode the final angle
        dimensions=1,
        label="Angle_Decoder",
        neuron_type=nengo.LIF(),  # Spiking neurons
        max_rates=nengo.dists.Uniform(100, 200),
    )

    # Connect on and off neurons to the angle decoder.
    # The decoder will learn to combine the on and off spike patterns
    # to produce a smooth angle estimate.
    nengo.Connection(on_neurons, angle_snn, synapse=0.01)
    nengo.Connection(off_neurons, angle_snn, synapse=0.01)

    # ========================================================================
    # OUTPUT LAYER: FINAL ANGLE IN DEGREES
    # ========================================================================
    # Convert the decoded angle (which may be in different units) to degrees.

    angle_output = nengo.Node(
        size_in=1,
        label="Angle_Output_deg"
    )

    # Connect the angle decoder to the output node.
    # The function=lambda applies the ILD-to-angle mapping.
    nengo.Connection(
        angle_snn,
        angle_output,
        function=lambda x: ild_to_angle(x[0]),
        synapse=0.05  # Smooth over 50 ms
    )

    # ========================================================================
    # VISUALIZATION HELPER: COMPASS DISPLAY
    # ========================================================================
    # For a nice visualization in Nengo GUI, convert the angle to (x, y)
    # coordinates on a unit circle. This lets us plot the angle as a
    # dot moving left/right (like a compass needle).

    def angle_to_xy(t, angle_deg):
        """
        Convert angle in degrees to unit circle coordinates (x, y).

        Parameters
        ----------
        t : float
            Simulation time (not used).
        angle_deg : np.ndarray
            Array containing the angle in degrees.

        Returns
        -------
        list
            [x, y] coordinates on a unit circle.
        """
        # Convert angle from degrees to radians.
        angle_rad = np.deg2rad(angle_deg[0])

        # Compute unit circle coordinates.
        # When angle = 0°, (x, y) = (1, 0) — point to the right.
        # When angle = 90°, (x, y) = (0, 1) — point up.
        # When angle = -90°, (x, y) = (0, -1) — point down.
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)

        return [x, y]

    # Node that computes the compass coordinates.
    angle_xy = nengo.Node(
        angle_to_xy,
        size_in=1,  # Receives the angle (1 input)
        size_out=2,  # Produces (x, y) (2 outputs)
        label="Angle_Compass"
    )

    # Connect the angle output to the compass node.
    nengo.Connection(angle_output, angle_xy, synapse=0.05)

    # ========================================================================
    # PROBES: RECORD DATA FOR ANALYSIS AND VISUALIZATION
    # ========================================================================
    # Probes are like "recording electrodes" that store data as the model runs.
    # You can view them in Nengo GUI in real time and analyze offline.

    # Record the raw ILD feature.
    p_ild = nengo.Probe(feat_node, synapse=None)

    # Record spikes from the ILD encoding layer (you can plot spike rasters).
    p_ild_spikes = nengo.Probe(ild_snn.neurons)

    # Record spikes from on and off neurons.
    p_on_spikes = nengo.Probe(on_neurons.neurons)
    p_off_spikes = nengo.Probe(off_neurons.neurons)

    # Record the decoded angle from the angle decoder ensemble (before mapping).
    p_angle_raw = nengo.Probe(angle_snn, synapse=None)

    # Record the final angle output in degrees.
    p_angle_output = nengo.Probe(angle_output, synapse=0.05)

    # Record the (x, y) compass coordinates for visualization.
    p_angle_xy = nengo.Probe(angle_xy, synapse=0.05)

# ============================================================================
# START THE AUDIO STREAM
# ============================================================================
# Begin recording from the microphones immediately so audio is available
# as soon as the simulation starts.

stream = start_stream(device=5, channels=2)
print(f"Spiking Neural Network Model Started")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"Timestep (DT): {DT:.4f} seconds")
print(f"In Nengo GUI:")
print(f"  - Right-click 'ILD_SpikeLayer' → Spikes (watch neurons fire!)")
print(f"  - Right-click 'Angle_Compass' → XY value (watch the compass)")
print(f"  - Right-click 'Angle_Output_deg' → Value (watch the angle)")

# ============================================================================
# OPTIONAL: STANDALONE TEST (WITHOUT NENGO GUI)
# ============================================================================
# If you run this file directly (not via nengo command), it will run the analyze_offline file and simulate for 10 seconds.

if __name__ == "__main__":
    # Import and run the offline analysis
    from analyze_offline import *
