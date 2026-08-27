# BlackHole-Neural-Simulator

A deep learning project that generates realistic black hole images from physical parameters in real time. Instead of performing computationally expensive ray tracing, a neural network learns the mapping between simulation parameters and rendered images, enabling interactive visualization at inference speed.

> Dataset generated using a modified version of the excellent project: https://github.com/kavan010/black_hole

## Highlights

- Trained a **PyTorch** neural network to generate black hole renders from astrophysical parameters.
- Built an **interactive simulator** with real-time parameter controls using Matplotlib.
- Replaced expensive ray tracing with a **surrogate neural network**, reducing image generation time from seconds to milliseconds.
- Supports live adjustment of:
  - Black hole mass
  - Camera elevation & azimuth
  - Accretion disk inner radius
  - Accretion disk outer radius

## Model

**Input Features**
- Black hole mass
- Schwarzschild radius
- Camera azimuth
- Camera elevation
- Camera distance
- Disk inner radius
- Disk outer radius

**Architecture**
- Fully Connected Layers
- Transposed Convolutional Decoder
- Output: **150 × 200 RGB Image**

**Training**
- Framework: PyTorch
- Optimizer: Adam
- Loss: L1 Loss

## Disclaimer

This project is a **neural approximation** of a black hole ray tracer. It is intended for fast visualization and machine learning experimentation rather than physically accurate scientific simulation.
