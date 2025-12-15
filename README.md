# 🎙️ VALL-E-X Voice Cloning System  
### _High-Fidelity Neural Text-to-Speech with Advanced Speaker Similarity_

---

## 🌟 **Project Overview**

**> 🚀 This project is built on top of an **unofficial Microsoft-released implementation of VALL-E-X**, enhanced with **additional features, quality improvements, and evaluation metrics**.**

The system delivers **high-quality voice cloning** with strong **speaker identity preservation** and **word-level accuracy**, making it suitable for advanced **text-to-speech research and experimentation**.

### ✨ **Key Highlights**

- 🧠 **Better voice cloning quality** with improved speaker similarity  
- 🎯 **High word-level accuracy metrics** for speech evaluation  
- 🗣️ **Speaker diarization** support  
- 🔍 **Audio validation & vocal similarity analysis**  
- ▶️ **Play & download options** for generated audio output  
- 🎵 Supports **`.wav` and `.mp3`** input audio formats  
- ⚡ Optimized inference pipeline for cleaner and more stable outputs  


---

## 🔧 **Installation & Setup**

### 📥 **1. Clone the Repository**


git clone <your_repo_url_here>

cd <your_repo_name>

### 🐍 **2. Python Requirements**


Python Version: 3.11

Recommended: Anaconda / Miniconda

## 📦 **3. Create & Activate Conda Environment (Recommended)**

### Install Anaconda (If Not Installed)

Download Anaconda from:  
👉 https://www.anaconda.com/products/distribution

During installation:
- ✅ **Check**: *Add Anaconda to PATH*
- ✅ **Check**: *Register Anaconda as default Python*

After installation, open **Anaconda Prompt** or terminal and verify:

conda --version

**Creating the anaconda environment (Put the command in cmd prompt)**

conda create -n <put_your_env_name> python=3.11

conda activate <put_your_env_name>

📌 Example:

conda create -n valle-env python=3.11

conda activate valle-env

**OR**

**🧪 Alternative: Python Virtual Environment (venv)**

If you do not want to use Conda, you can use Python’s built-in virtual environment.

🔹 Step 1: Ensure Python 3.11 is Installed

python --version

If not installed, download from:

👉 https://www.python.org/downloads/

⚠️ Make sure Python is added to PATH during installation

🔹 Step 2: Create Virtual Environment

python -m venv venv

🔹 Step 3: Activate Virtual Environment

Windows

venv\Scripts\activate


macOS / Linux

source venv/bin/activate


✅ After activation, you should see (venv) in your terminal.

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

**RUN**

python launch-ui.py


🎙️ Input Audio Requirements

Supported formats: .wav, .mp3

Mono channel preferred

6-10 seconds of clean reference audio works best

500 - 600 characters text works best

📤 Output

The system generates:

🔊 Synthesized speech (.wav)

📊 Word accuracy metrics

🧠 Speaker similarity scores

▶️ Audio playback & download options


## ⚠️ Troubleshooting & Notes

> **Note:** If the system encounters an unknown or transient error, restarting the system is recommended, as it often resolves the issue.

- Ensure FFmpeg is correctly installed and added to PATH  
- Make sure the correct Conda / virtual environment is activated  
- Close and relaunch the UI if audio playback fails  
- Restart the system before deeper debugging if unexpected errors occur
  

### **📁 Project Structure**

```text
├── customs/               # Custom user-defined components & overrides
├── data/                  # Dataset files and intermediate data
├── images/                # Images used for UI / documentation
├── models/                # Core model architectures
├── modules/               # Modularized model & pipeline components
├── nltk_data/             # NLTK resources required for text processing
├── presets/               # Predefined configuration presets
├── prompts/               # Prompt templates for inference
├── utils/                 # Utility functions and helpers              
├── descriptions.py        # Model / feature descriptions
├── examples.py            # Example usage scripts
├── launch-ui.py           # Main entry point to launch UI & inference
├── macros.py              # Global macros and constants
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```


