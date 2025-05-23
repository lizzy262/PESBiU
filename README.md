# PESBiU: Predictive Energy Saving for Baseband Units

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
  - [PESBiU 1.0](#pesbiu-10)
  - [PESBiU 2.0](#pesbiu-20)
- [Architecture](#architecture)
- [Evaluation](#evaluation)


## Introduction

**PESBiU** (Predictive Energy Saving for Baseband Units) is an ML-driven algorithm designed to optimize energy consumption in Baseband Units (BBUs) within 5G networks. PESBiU addresses the critical need for energy efficiency in the face of increasing data traffic and the demand for sustainable network operations.

This repository contains two versions of the PESBiU algorithm:

- **PESBiU 1.0**: The initial version focusing on traffic volume prediction and threshold-based BBU management.
- **PESBiU 2.0**: An enhanced version incorporating advanced reinforcement learning techniques for more intelligent and adaptive BBU management.

## Features

### PESBiU 1.0

- **Traffic Volume Prediction**: Utilizes Long Short-Term Memory (LSTM) models to forecast traffic volumes in hourly intervals.
- **Threshold-Based Management**: Proactively transitions BBUs into sleep mode based on predicted traffic thresholds to save energy.
- **Real-World Data Utilization**: Leverages actual network data to validate the correlation between traffic, energy usage, and user experience.
- **Energy Savings**: Demonstrated a 15.12% reduction in energy consumption for a 24-hour period in simulations.

### PESBiU 2.0

- **Granular Data Handling**: Processes 15-minute interval data using Hyper-CNN-LSTM models for more accurate short-term predictions, which are available here: [Stage1_PESBiU](https://github.com/vafekt/Stage1_PESBiU.git)
- **Advanced Metrics**: Incorporates Total Port Throughput (Mbps), UE DL Latency (ms), and BBU AVG Power Consumption (W) for comprehensive monitoring.
- **Reinforcement Learning Integration**: Employs multi-agent reinforcement learning (DQN, DDDQN, A2C and Hybrid version) to make intelligent decisions on BBU sleep modes.
- **Safe Switch Mechanism**: Ensures network stability by activating all BBUs if significant discrepancies occur between predicted and actual traffic.
- **Multi-Site Management**: Supports multiple cell sites (Office, Residential, Remote) with transfer learning capabilities to streamline agent training.

## Architecture

The PESBiU system architecture is structured as follows corresponding to its version:

![PESBiU Architecture](assets/pesbiu1vs2.drawio2.png) 

## Evaluation

### PESBiU 1.0

- **Simulation Setup**: Implemented on a standard PC using LSTM predictions.
- **Results**: Achieved a 15.12% reduction in energy consumption over a 24-hour period by strategically putting BBUs to sleep during low traffic periods.

### PESBiU 2.0

- **Simulation Setup**: Utilized a high-performance server with NVIDIA Tesla V100S-PCIE-32GB GPUs and CUDA 12.3 for training reinforcement learning agents.
- **Results**: Enhanced energy savings with improved accuracy in traffic prediction and adaptive BBU management through reinforcement learning. The evaluations demonstrate that PESBiU 2.0 can reduce BBU power consumption by up to 40% without
compromising service quality. Precisely, we achieved 41.21% energy saving over a 24-hour period with DDDQN.




