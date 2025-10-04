# EH5TinyML

📖 Introduction

This project is developed as part of the Tiny Machine Learning course.
The goal is to design an embedded artefact on the Particle Photon 2, which can recognize whether an engine at idle is running normally or with a faulty ignition system using a TinyML model.

🎯 Purpose

The purpose is to demonstrate how embedded hardware, combined with audio preprocessing and machine learning, can be used to classify idle engine sounds in real time.
This could potentially be used in diagnostics and maintenance of vehicles, for quick identification of ignition-related issues.

🛠️ System Description

Platform: Particle Photon 2 (ARM Cortex-M33)

Sensor: PDM microphone for capturing engine audio

Communication: TCP transfer to laptop for data logging

Preprocessing:

Audio segmentation

feature extraction

Model: TinyML classification model (normal vs. faulty idle)


✅ Functional Requirements

Capture engine audio data via microphone

Collect and store labeled dataset (normal / faulty ignition)

Preprocess audio into features

Train ML model and deploy on Photon 2

Output prediction results in real time (normal / faulty)

📊 Dataset

Audio recordings from multiple cars at idle:

CAR_XX_NORMAL → 5 minutes of normal idle sound

CAR_XX_FAULTY → 5 minutes of faulty idle sound (Removed ignition coil on one cylinder)

Dataset collected with permission from KP Biler / Mikkel O. Pedersen Racing

⚙️ Implementation Steps

Set up PDM microphone on Particle Photon 2 and verify audio capture.

Record engine idle sounds (normal and faulty).

label dataset.

examine dataset, find most relevant features(
Time domain: RMS energy, Amplitude envelope, Zero-crossing rate.
Time frequency domain: STFT.
frequency domain: Spectral analysis, MFCC.

Train ML model.

Convert model and deploy to Photon 2.

Verify real-time classification performance.

🧪 Test & Verification

Validation of dataset with split (train/test)

Accuracy evaluation of trained ML model.

Real-world test: connect Photon 2 to microphone near an idling engine and see classification output.

👤 Author

Mads O. Pedersen
