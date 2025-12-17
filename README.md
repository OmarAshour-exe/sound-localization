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

## ⚙️ Architecture

to be added... 

A simplified diagram:

```
[ Mic 1 ]     [ Mic 2 ]
     ↓            ↓
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
python main.py
```

### 4. Connect external devices
Connect iPhone or iPad as additional microphones via AirPlay, Bluetooth, or wired input for spatial recording tests.

---

## 📊 Results

- Plots display spike activity and estimated sound angles.  
- Model performance is evaluated by comparing predicted vs. true source locations.  
- Demonstrates feasibility of low-power, biologically plausible auditory localization.

---

## 📁 Project Structure

```
├── main.py                # Entry point for the Nengo model
├── localization_model.py  # SNN architecture and simulation setup
├── audio_input.py         # Microphone synchronization and recording
├── utils/                 # Signal processing helper functions
├── data/                  # Sample recordings and testing data
└── README.md
```

---

## 🎓 Contributors

Developed as part of the **Neuromorphic Intelligence** course at **Technische Hochschule Nürnberg**.

**Team Members:** Omar Ashour & Zaid Mahayni

---

## 📜 License

This project is for **educational and research purposes**.  
Feel free to fork or adapt it with credit to the original authors.