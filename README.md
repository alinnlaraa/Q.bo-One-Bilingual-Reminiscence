# Q.bo One Embodied Conversational AI System for Reminiscence

This project was build as part of a Master Thesis at the Technical University of Vienna.
This repository contains the implementation of an embodied conversational AI system using the Q.bo One robot. The system integrates a local large language model (LLM), speech-to-text (STT), and text-to-speech (TTS) components in a distributed architecture consisting of a robot and an external workstation.

## Overview

The system enables natural spoken interaction with a Q.bo One robot by combining:

- **Local LLM inference** via Ollama
- **Speech-to-text (STT)** using Faster-Whisper
- **Text-to-speech (TTS)** using Piper
- **Robot control layer** (LEDs, audio playback, communication)
- **External AI pipeline** running on a workstation (GPU-enabled)

The robot acts as an embodied interface, while the computationally intensive AI components run on an external PC to ensure performance and flexibility.

## System Architecture

The system is divided into two main components:

### 1. Q.bo One Robot (Edge Device)
- Audio playback (ALSA-based)
- LED status control (via serial interface)
- TCP communication with workstation
- Lightweight client scripts

### 2. External Workstation (AI Brain)
- Runs Ollama-based LLM inference
- Processes speech recognition (Faster-Whisper)
- Generates speech output (Piper TTS)
- Manages conversation state and memory
- Handles dialogue orchestration

## Key Features

- Fully local and privacy-preserving architecture (no cloud dependency)
- Multilingual dialogue system (German & Turkish)
- Persistent conversation memory (SQLite-based)
- Real-time voice interaction
- Robot status visualization via LED feedback
- Modular pipeline design (STT → LLM → TTS)

## Hardware Setup

- Q.bo One robot (Raspberry Pi 3)
- External workstation (GPU-enabled)
- USB microphone (AirHug 21)
- Network connection (LAN/WiFi)

## Software Stack

- Python 3.12 (workstation)
- Python 2.7 / 3.5 (robot legacy OS)
- Ollama (LLM backend)
- Faster-Whisper (STT)
- Piper TTS (speech synthesis)
- LangChain / LangGraph (conversation management)
- Ubuntu 24.04 LTS (workstation)

## How It Works

1. Microphone input is captured on the workstation
2. Speech is transcribed using Faster-Whisper
3. Input is sent to a local LLM (Ollama)
4. Response is generated and processed
5. Piper converts text to speech
6. Audio is streamed to the robot
7. Robot plays audio and updates LED status

## Purpose

This project explores the integration of large language models with embodied robotic systems in a privacy-preserving, fully local setup, focusing on conversational interaction for assistive and social robotics applications.

## License

Apache License Version 2.0
