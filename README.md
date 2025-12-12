VALL-E Clone – Text-to-Speech Pipeline

This repository contains an implementation of VALL-E, a neural codec–based text-to-speech system that generates speech conditioned on a short audio prompt.

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

