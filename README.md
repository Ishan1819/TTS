# 🎙️ VALL-E-X Voice Cloning System  
### _High-Fidelity Neural Text-to-Speech with Advanced Speaker Similarity_

---

## 🌟 **Project Overview**

> 🚀 This project is built on top of an **unofficial Microsoft-released implementation of VALL-E-X**, enhanced with **additional features, quality improvements, and evaluation metrics**.

The system delivers **high-quality voice cloning** with strong **speaker identity preservation** and **word-level accuracy**, making it suitable for advanced **text-to-speech research and experimentation**.

### ✨ **Key Highlights**

- 🧠 **Better voice cloning quality** with improved speaker similarity  
- 🎯 **High word-level accuracy metrics** for speech evaluation  
- 🗣️ **Speaker diarization** support  
- 🔍 **Audio validation & vocal similarity analysis**  
- ▶️ **Play & download options** for generated audio output  
- 🎵 Supports **`.wav` and `.mp3`** input audio formats  
- ⚡ Optimized inference pipeline for cleaner and more stable outputs  

> ⚠️ This repository is intended for **research, experimentation, and educational purposes only**.

---

## 🔧 **Installation & Setup**

### 📥 **1. Clone the Repository**


git clone <your_repo_url_here>
cd <your_repo_name>

### 🐍 **2. Python Requirements**


Python Version: 3.11

Recommended: Anaconda / Miniconda

### 📦 **3. Create & Activate Conda Environment**


conda create -n <put_your_env_name> python=3.11

conda activate <put_your_env_name>

📌 Example:

conda create -n valle-env python=3.11

conda activate valle-env


### 🎵 **4. Install FFmpeg (Required for recording of audio)**

FFmpeg is required for audio processing and format handling.

🪟 Windows

Download from:

👉 https://www.gyan.dev/ffmpeg/builds/

ffmpeg-2025-12-07-git-c4d22f2d2c-full_build.7z

After downloading:

Extract the archive

Add the bin/ folder to your System PATH

🍎 macOS

brew install ffmpeg

🐧 Linux

sudo apt update

sudo apt install ffmpeg

✅ Verify installation:

ffmpeg -version


### 📥 **5. Install Python Dependencies**

Make sure your environment is activated, then run:

pip install -r requirements.txt

### ▶️ **6. Running the Model**

Example inference command:

python launch-ui.py

🎙️ Input Audio Requirements

Supported formats: .wav, .mp3

Mono channel preferred

6-10 seconds of clean reference speech works best

📤 Output

The system generates:

🔊 Synthesized speech (.wav)

📊 Word accuracy metrics

🧠 Speaker similarity scores

▶️ Audio playback & download options

📁 Project Structure

