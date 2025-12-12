VALL-E X – Voice Cloning & TTS System

This repository contains an implementation of VALL-E, a neural codec–based text-to-speech system that generates speech conditioned on a short audio prompt.
A high-quality voice cloning and text-to-speech (TTS) system built with VALL-E architecture, enhanced with additional validation, accuracy metrics, and improved robustness.

🚀 Features Added (Custom Enhancements)
✔ 1. Input Character Length Limiter

Prevents the user from entering overly long text that may break TTS or produce unstable output.
System automatically validates input length before generating audio.

✔ 2. TTS Accuracy Metric

A custom metric that evaluates the quality of generated speech compared to model expectations.

✔ 3. Voice-Cloning Accuracy Score

Measures similarity between cloned voice and original reference using audio embeddings.

✔ 4. Multi-Voice Detection Safety Check

If more than 2 voices are detected in the reference audio, the model raises an error:

❌ "Multiple speakers detected. Please upload clean single-speaker audio."

This ensures clean, high-accuracy cloning.
🔧 Setup Instructions
1. Clone the Repository
git clone <your_repo_url_here>
cd <your_repo_name>


Replace the URL with your actual GitHub repository link.

🐍 2. Python Version

This project requires:

Python 3.11


Make sure you have Anaconda installed (recommended).

📦 3. Create & Activate Conda Environment
conda create -n <your_env_name> python=3.11
conda activate <your_env_name>


Replace <your_env_name> with your desired environment name (example: valle-env).

🎵 4. Install FFmpeg

FFmpeg is required for handling audio files.

Windows

Download from:
https://www.gyan.dev/ffmpeg/builds/

Add FFmpeg's bin/ folder to your system PATH.

macOS
brew install ffmpeg

Linux
sudo apt update
sudo apt install ffmpeg


To verify installation:

ffmpeg -version

📥 5. Install Dependencies

After activating your environment, run:

pip install -r requirements.txt


