# Neuromorphic Sound Localization

This project implements **sound source localization** using **neuromorphic computing principles**. We simulate a biologically inspired auditory system with **spiking neural networks (SNNs)** in **Nengo**, processing real-time or recorded audio captured from **MacBook, iPhone, and iPad microphones**.

---

## 🧠 Background Theory

In biological audition, when a sound reaches one ear a bit earlier than the other, the signal from that ear starts travelling along its nerve fibers slightly sooner. The signal from the later ear has less distance to travel, so the two signals meet at different positions along a row of special neurons that only respond when inputs from both ears arrive at the same time. Each neuron in this row therefore becomes active for one particular time difference between the ears, and together they form a kind of internal ruler that tells the brain from which direction the sound is coming.

![Auditory Spatial Mapping](https://github.com/OmarAshour-exe/sound-localization/blob/master/assets/Auditory_Spatial_Mapping.png)

The sound itself is turned into short “on” and “off” spikes: one group of neurons fires when the sound level goes up, another when it goes down. By adding up these spikes with simple filters, the brain—or a neuromorphic model—can reconstruct a smooth version of the original signal, while still enjoying the robustness and efficiency of spike-based processing.

![On And Off spikes](https://github.com/OmarAshour-exe/sound-localization/blob/master/assets/On_Off_spikes.png)

---

## 🗺️ Overview

Human hearing uses interaural time and level differences to estimate the direction of sound sources. This project replicates that mechanism by:

- Capturing audio from two spatially separated microphones.  
- Computing interaural time and intensity differences in real-time.  
- Using a spiking neural network built with **Nengo** to estimate the sound source direction.  

The system demonstrates how **neuromorphic architectures** can process sensory signals efficiently and robustly.

---

## Design Choices

### Why ILD instead of ITD?

**Hardware Synchronization Constraints:**  
ITD relies on microsecond-level timing differences between signals arriving at the two ears. Our setup uses two independent devices (MacBook and iPhone) connected via software. The latency and clock synchronization between these distinct devices are not stable enough to reliably measure the tiny phase differences required for ITD.

In contrast, **ILD (Interaural Level Difference)** measures relative loudness. It is robust against small timing jitters and latency variations, making it far more reliable for a distributed microphone array while still effectively demonstrating neuromorphic processing principles.

---

## ⚙️ Architecture

The system is designed as a pipeline that converts raw audio into spiking activity and finally into a direction estimate.

1.  **Audio Input Layer (Preprocessing):**
    - Captures live audio streams from two devices (e.g., Mac and iPhone).
    - Normalizes the gain of each microphone to account for hardware differences.
    - Computes the **Interaural Level Difference (ILD)** in decibels (dB) for each 16ms audio frame.

2.  **Spiking Encoding Layer (`ild_snn`):**
    - The continuous ILD value is fed into a population of **Leaky Integrate-and-Fire (LIF)** neurons.
    - These neurons encode the scalar ILD value into a temporal pattern of spikes, similar to how the auditory nerve encodes sound intensity.

3.  **On/Off Neuron Processing (`on_neurons` & `off_neurons`):**
    - The signal splits into two specialized populations:
        - **On-neurons:** Respond primarily when the signal shifts towards the Mac side (positive ILD).
        - **Off-neurons:** Respond primarily when the signal shifts towards the iPhone side (negative ILD).
    - This mimics biological "On" and "Off" channels in the auditory pathway, increasing robustness and dynamic range.

4.  **Angle Decoding Layer (`angle_snn`):**
    - A final population of neurons receives spike input from both On and Off populations.
    - It integrates these signals to decode a stable estimate of the sound source angle.

5.  **Output Visualization:**
    - The decoded angle is mapped to degrees (-90° to +90°) and visualized as a compass needle in real-time.

**Simplified Data Flow:**

```
[ Mic 1 (Mac) ] [ Mic 2 (iPhone) ]
       ↓                ↓
[ Feature Extraction (Compute ILD) ]
               ↓
[ Spiking Encoder Layer (LIF Neurons) ]
         ↙          ↘
[ On-Neurons ] [ Off-Neurons ]
         ↘          ↙
[ Angle Decoder Layer (Integration) ]
               ↓
[ Final Direction Estimate (Degrees) ]
```

---

## 🧩 Technologies

- **Language:** Python 3  
- **Framework:** [Nengo](https://www.nengo.ai/)  
- **Devices:** MacBook, iPhone, iPad (for microphone input)  
- **Dependencies:** numpy, sounddevice / avfoundation APIs, matplotlib  

---

## 🚀 Getting Started

### 1. Clone the repository
```
git clone https://github.com/OmarAshour-exe/sound-localization.git
cd neuromorphic-sound-localization
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the Nengo model
```
nengo main.py
```
*Note: This starts the audio stream and opens the Nengo GUI for visualization.*

### 4. Connect external devices
Ensure your iPhone/iPad is connected as an audio input device (e.g., via Audio MIDI Setup on macOS) before running the script.

---

## To-Do
Add something about analyze_offile and live_detection.

---

## 📊 Results

- **Real-time Spike Rasters:** Visualizes the firing patterns of On and Off neurons as they respond to sound direction.
- **Compass Plot:** A live XY-plot shows the estimated direction of the sound source relative to the microphones.
- **Robustness:** The SNN successfully tracks a moving speaker even with background noise, demonstrating the stability of the rate-coding approach.

---

## 📁 Project Structure

```
├── main.py             # Entry point: Nengo model, SNN definition, and GUI setup
├── audio_input.py      # Handles live audio streaming and queue management
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🎓 Contributors

Developed as part of the **Neuromorphic Intelligence** course at **Technische Hochschule Nürnberg**.

**Team Members:** Omar Ashour & Zaid Mahayni

---

## 📜 License

This project is for **educational and research purposes**.  
Feel free to fork or adapt it with credit to the original authors.