import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
if(sys.version_info[0]<3 or sys.version_info[1]<7):
    print("The Python version is too low and may cause problems")

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import langid
langid.set_languages(['en'])
import nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

import re  # Add regex import at module level

import torch
import torchaudio
import random
import soundfile as sf

import numpy as np

from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *

import gradio as gr
import whisper
from vocos import Vocos
import multiprocessing
import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
import re
# ------------------------------
# Simple KMeans Speaker Diarization
# ------------------------------
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

import time
import os
import numpy as np

# Lazy imports for embedding & ASR evaluation
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import torchaudio
except Exception:
    torchaudio = None

# transformers Wav2Vec2 for speaker embeddings (lazy loaded)
_w2v_processor = None
_w2v_model = None
def _ensure_w2v_loaded():
    global _w2v_processor, _w2v_model
    if _w2v_processor is None or _w2v_model is None:
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            _w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            _w2v_model.eval()
            for p in _w2v_model.parameters():
                p.requires_grad = False
        except Exception as e:
            _w2v_processor = None
            _w2v_model = None
            print(f"[Eval Helper] Failed to load Wav2Vec2 model: {e}")

def clean_text_for_comparison(text):
    """Clean and normalize text for WER calculation"""
    text = re.sub(r'<\|[^|]+\|>', '', text)  # Remove <|en|> tokens
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def get_embedding(wav_path):
    import torch
    global _w2v_processor, _w2v_model
    if sf is None:
        raise RuntimeError("soundfile (pysoundfile) is required for get_embedding")
    _ensure_w2v_loaded()
    if _w2v_processor is None or _w2v_model is None:
        raise RuntimeError("Wav2Vec2 processor/model not loaded")
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    if sr != 16000:
        if torchaudio is None:
            raise RuntimeError("torchaudio required for resampling to 16k")
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav = wav_t.squeeze(0).numpy()
        sr = 16000
    inputs = _w2v_processor(wav, sampling_rate=sr, return_tensors="pt", padding=False)
    input_values = inputs["input_values"]
    if input_values.dim() == 3:
        input_values = input_values.squeeze(1)
    with torch.no_grad():
        outputs = _w2v_model(input_values)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return emb
     
     
def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="output"):
    """Enhanced evaluation metrics with speaker similarity"""
    import jiwer
    import whisper
    from scipy.io.wavfile import write as write_wav
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Save audio
    audio = np.asarray(audio_numpy, dtype=np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak

    timestamp = int(time.time())
    out_filename = f"{save_prefix}_{timestamp}.wav"
    
    # Save with proper format
    try:
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        write_wav(out_filename, sample_rate, audio_int16)
        print(f"[Debug] Saved generated audio to: {out_filename}")
    except Exception as e:
        print(f"[Error] Failed to save audio: {e}")
        return

    # Transcribe
    try:
        global whisper_model
        res = whisper_model.transcribe(out_filename)
        transcribed = res.get("text", "").strip()
    except:
        transcribed = "[ERROR]"
    
    # Calculate WER (but don't show Word Match if you don't want it)
    wer_val = None
    word_acc = None
    if original_text:
        clean_orig = clean_text_for_comparison(original_text)
        clean_trans = clean_text_for_comparison(transcribed)
        
        if clean_orig and clean_trans:
            try:
                wer_val = jiwer.wer(clean_orig, clean_trans)
                
                # Word accuracy (calculate but don't display if not needed)
                orig_words = set(clean_orig.split())
                trans_words = set(clean_trans.split())
                if len(orig_words) > 0:
                    word_acc = len(orig_words & trans_words) / len(orig_words)
            except Exception as e:
                print(f"[Error] WER calculation failed: {e}")
    
    # Speaker similarity (cloning ability)
    similarity = None
    cloning_quality = None
    
    print(f"[Debug] Reference path provided: {reference_prompt_wav_path}")
    
    if reference_prompt_wav_path:
        if not os.path.exists(reference_prompt_wav_path):
            print(f"[Warning] Reference audio not found: {reference_prompt_wav_path}")
        elif not os.path.exists(out_filename):
            print(f"[Warning] Generated audio not found: {out_filename}")
        else:
            try:
                print(f"[Debug] Computing embeddings...")
                print(f"[Debug] Reference file: {reference_prompt_wav_path}")
                print(f"[Debug] Generated file: {out_filename}")
                
                # Get embeddings
                emb_gen = get_embedding(out_filename)
                print(f"[Debug] Generated embedding shape: {emb_gen.shape}")
                
                emb_ref = get_embedding(reference_prompt_wav_path)
                print(f"[Debug] Reference embedding shape: {emb_ref.shape}")
                
                # Calculate cosine similarity
                emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
                emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
                similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
                
                print(f"[Debug] Similarity computed: {similarity:.4f}")
                
                # Determine cloning quality
                if similarity >= 0.80:
                    cloning_quality = "⭐⭐⭐⭐⭐ Excellent"
                elif similarity >= 0.70:
                    cloning_quality = "⭐⭐⭐⭐ Very Good"
                elif similarity >= 0.60:
                    cloning_quality = "⭐⭐⭐ Good"
                elif similarity >= 0.50:
                    cloning_quality = "⭐⭐ Average"
                else:
                    cloning_quality = "⭐ Poor"
                    
            except Exception as e:
                print(f"[Error] Speaker similarity calculation failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"[Debug] No reference path provided - skipping similarity")
    
    # Print results
    print("\n" + "="*60)
    print("            🎯 ACCURACY METRICS")
    print("="*60)
    print(f"Original     : {original_text}")
    print(f"Transcribed  : {transcribed}")
    
    if wer_val is not None:
        print(f"WER          : {wer_val:.4f} Word match error %")
    else:
        print(f"WER          : N/A")
    
    # Remove Word Match display if you don't want it
    # if word_acc is not None:
    #     print(f"Word Match   : {word_acc*100:.1f}%")
    
    # Speaker similarity section
    if similarity is not None:
        print(f"Similarity   : {similarity:.4f}")
        print(f"Cloning      : {cloning_quality}")
    else:
        print(f"Similarity   : N/A (check debug messages above)")
        print(f"Cloning      : Not evaluated")
    
    print(f"File         : {out_filename}")
    print("="*60 + "\n")
     
     
def detect_speakers(audio_path, threshold=0.15):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        if duration < 1.0:
            return 1, "✓ Single speaker (audio too short for analysis)"
        
        energy = librosa.feature.rms(y=y)[0]
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        
        voiced_frames = []
        for i in range(min(frames.shape[1], len(energy))):
            if energy[i] > np.mean(energy) * 0.8:
                voiced_frames.append(frames[:, i])
        
        if len(voiced_frames) < 5:
            return 1, "✓ Single speaker (insufficient voice activity)"
        
        embeddings = []
        for frame in voiced_frames:
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20)
            emb = np.mean(mfcc, axis=1)
            embeddings.append(emb)
        
        if len(embeddings) < 10:
            return 1, "✓ Single speaker (insufficient features)"
        
        embeddings = np.array(embeddings)
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
        labels = kmeans.labels_
        
        cluster_0 = embeddings[labels == 0]
        cluster_1 = embeddings[labels == 1]
        
        if len(cluster_0) == 0 or len(cluster_1) == 0:
            return 1, "✓ Single speaker detected"
        
        centroid_0 = cluster_0.mean(axis=0)
        centroid_1 = cluster_1.mean(axis=0)
        
        distance = cosine_distances([centroid_0], [centroid_1])[0][0]
        
        cluster_0_pct = len(cluster_0) / len(embeddings) * 100
        cluster_1_pct = len(cluster_1) / len(embeddings) * 100
        
        print(f"Cluster analysis: {cluster_0_pct:.1f}% vs {cluster_1_pct:.1f}%, distance={distance:.3f}")
        
        if distance < threshold:
            return 1, f"✓ Single speaker detected (distance={distance:.3f})"
        else:
            min_cluster_size = 15
            if cluster_0_pct < min_cluster_size or cluster_1_pct < min_cluster_size:
                return 1, f"✓ Single speaker (minor variation detected, distance={distance:.3f})"
            
            return 2, f"⚠ Multiple speakers detected ({cluster_0_pct:.0f}%/{cluster_1_pct:.0f}% split, distance={distance:.3f})"
    
    except Exception as e:
        print(f"Speaker detection error: {e}")
        return 1, "✓ Single speaker (analysis failed, assuming single)"

def check_audio_for_generation(audio_path):
    num_speakers, detail_msg = detect_speakers(audio_path)
    
    if num_speakers > 1:
        error_msg = (
            "❌ Multiple speakers detected in the audio!\n\n"
            "This voice cloning system requires audio with only ONE speaker.\n"
            "Please provide a different audio sample with only one person speaking."
        )
        return False, error_msg
    else:
        success_msg = f"✓ Audio validated for voice cloning\n{detail_msg}"
        return True, success_msg

thread_count = multiprocessing.cpu_count()
print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")

if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
CHECKPOINT_PATH = "checkpoints/vallex-checkpoint.pt"

os.makedirs("checkpoints", exist_ok=True)

# Create prompts directory if it doesn't exist
if not os.path.exists("./prompts/"): 
    os.mkdir("./prompts/")

if not os.path.isfile(CHECKPOINT_PATH):
    import wget
    try:
        print("Model checkpoint not found. Downloading it now...")
        logging.info("Downloading VALLE-X model (first time only)...")
        wget.download(
            "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
            out=CHECKPOINT_PATH,
            bar=wget.bar_adaptive
        )
        print("\nDownload complete!")
    except Exception as e:
        logging.info(e)
        raise Exception(
            "\nModel weights download failed.\n"
            "Please manually download from https://huggingface.co/Plachta/VALL-E-X\n"
            f"and put vallex-checkpoint.pt inside: {os.getcwd()}/checkpoints/"
        )
else:
    print("✔ Using existing model checkpoint — skipping download.")

model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    )
checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu', weights_only=False)
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.eval()

audio_tokenizer = AudioTokenizer(device)
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

preset_list = os.walk("./presets/").__next__()[2]
preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

speaker_diarization_model = None

def load_speaker_diarization():
    print("Speaker detection will be skipped")

def check_single_speaker(audio_path):
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        if duration < 1.0:
            return True, 1, ""
        
        return True, 1, ""
        
    except Exception as e:
        print(f"Warning: Speaker diarization check failed: {e}")
        return True, 1, ""

def clear_prompts():
    """Clean up old temporary files"""
    try:
        path = tempfile.gettempdir()
        for eachfile in os.listdir(path):
            filename = os.path.join(path, eachfile)
            if os.path.isfile(filename) and filename.endswith(".npz"):
                lastmodifytime = os.stat(filename).st_mtime
                endfiletime = time.time() - 60
                if endfiletime > lastmodifytime:
                    os.remove(filename)
    except:
        pass
    
    # Also clean up temp enrollment audio files in current directory
    try:
        for eachfile in os.listdir(os.getcwd()):
            if eachfile.startswith("temp_enroll_") and eachfile.endswith(".wav"):
                filename = os.path.join(os.getcwd(), eachfile)
                if os.path.isfile(filename):
                    lastmodifytime = os.stat(filename).st_mtime
                    endfiletime = time.time() - 300  # Delete files older than 5 minutes
                    if endfiletime > lastmodifytime:
                        os.remove(filename)
                        print(f"[Cleanup] Deleted old temp file: {eachfile}")
    except:
        pass

def transcribe_one(model, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)
    print(result.text)
    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr

def make_prompt(name, wav, sr, save=True):
    global whisper_model
    whisper_model.to(device)
    
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    
    if wav.abs().max() > 1:
        wav = wav / wav.abs().max()
    
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    
    assert wav.ndim == 2 and wav.size(0) == 1, f"Expected shape (1, N), got {wav.shape}"
    
    data = wav.squeeze(0).cpu().numpy()
    data = np.clip(data, -1.0, 1.0)
    data = data.astype(np.float32)
    data = np.clip(data, -1.0, 1.0)
    data_int16 = (data * 32767).astype(np.int16)
    sf.write(f"./prompts/{name}.wav", data_int16, sr, subtype='PCM_16')
    
    lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    
    if lang != "en":
        raise ValueError(f"Error: Only English audio is supported. Detected language: {lang}")
    
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    
    return text, lang

def split_long_text(text, max_length=1000):
    """Split long text into chunks at sentence boundaries"""
    # Extract language token if present
    lang_token = ""
    clean_text = text
    
    if text.startswith("<|") and "|>" in text[:10]:
        end_idx = text.index("|>") + 2
        lang_token = text[:end_idx]
        clean_text = text[end_idx:]
        
        # Remove trailing language token if present
        if clean_text.endswith(lang_token):
            clean_text = clean_text[:-len(lang_token)]
    
    clean_text = clean_text.strip()
    
    # If text is short enough, return as single chunk
    if len(clean_text) <= max_length:
        if lang_token:
            return [lang_token + clean_text + lang_token]
        return [clean_text]
    
    # Split into sentences using multiple delimiters
    # This regex keeps the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        # Fallback: if no sentence boundaries, split by max_length
        chunks = []
        for i in range(0, len(clean_text), max_length):
            chunk = clean_text[i:i + max_length]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        # Add language tokens back
        if lang_token:
            chunks = [lang_token + chunk + lang_token for chunk in chunks]
        
        return chunks if chunks else [text]
    
    # Combine sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is longer than max_length, split it by words
        if len(sentence) > max_length:
            # First, save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long sentence by words
            words = sentence.split()
            temp_sentence = ""
            
            for word in words:
                if len(temp_sentence) + len(word) + 1 <= max_length:
                    temp_sentence += word + " "
                else:
                    if temp_sentence:
                        chunks.append(temp_sentence.strip())
                    temp_sentence = word + " "
            
            if temp_sentence:
                current_chunk = temp_sentence.strip()
        else:
            # Check if adding this sentence exceeds max_length
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out any empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    # If no valid chunks were created, return original text
    if not chunks:
        return [text]
    
    # Add language tokens back to each chunk
    if lang_token:
        chunks = [lang_token + chunk + lang_token for chunk in chunks]
    
    return chunks

@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt):
    global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
    try:
        clear_prompts()
        
        audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt
        if audio_prompt is None:
            return "No audio provided. Please provide an audio clip", None

        # Handle audio input
        temp_wav_path = None
        sr = None
        wav_pr = None

        if isinstance(audio_prompt, str):
            wav_pr, sr = torchaudio.load(audio_prompt)
            temp_wav_path = audio_prompt
        elif isinstance(audio_prompt, dict):
            audio_path = audio_prompt.get('name') or audio_prompt.get('path')
            if audio_path and os.path.isfile(audio_path):
                wav_pr, sr = torchaudio.load(audio_path)
                temp_wav_path = audio_path
            elif 'array' in audio_prompt:
                arr = audio_prompt['array']
                if isinstance(arr, (list, tuple)) and len(arr) == 2:
                    sr, data = arr
                    wav_pr = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
                elif isinstance(arr, np.ndarray):
                    wav_pr = torch.FloatTensor(arr)
                    sr = audio_prompt.get('sample_rate', 24000)
        elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
            sr, wav_pr = audio_prompt
            if not isinstance(wav_pr, torch.Tensor):
                wav_pr = torch.FloatTensor(wav_pr)
        
        # Save temp file for similarity
        if temp_wav_path is None or not os.path.isfile(temp_wav_path):
            tmp_path = f"temp_enroll_{int(time.time())}.wav"
            wav_save = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
            wav_save = np.clip(wav_save, -1.0, 1.0)
            sf.write(tmp_path, wav_save, sr, subtype='PCM_16')
            temp_wav_path = tmp_path

        # Validate audio
        if temp_wav_path:
            is_valid, check_msg = check_audio_for_generation(temp_wav_path)
            if not is_valid:
                return check_msg, None

        # Normalize audio
        if not isinstance(wav_pr, torch.Tensor):
            wav_pr = torch.FloatTensor(wav_pr)
        if wav_pr.abs().max() > 1:
            wav_pr = wav_pr / wav_pr.abs().max()
        if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
            wav_pr = wav_pr.mean(dim=0, keepdim=True)
        elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
            wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)
        if wav_pr.ndim == 1:
            wav_pr = wav_pr.unsqueeze(0)

        # Create prompt
        text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)

        # Prepare text
        if language == 'auto-detect':
            lang_token = lang2token[langid.classify(text)[0]]
        else:
            lang_token = langdropdown2token[language]
        lang = token2lang[lang_token]
        text = lang_token + text + lang_token

        # SPLIT LONG TEXT INTO CHUNKS
        text_chunks = split_long_text(text, max_length=1000)
        print(f"[Info] Split into {len(text_chunks)} chunk(s)")
        
        model.to(device)
        encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
        audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

        all_audio_chunks = []
        
        # Generate each chunk
        for idx, text_chunk in enumerate(text_chunks):
            print(f"[Chunk {idx+1}/{len(text_chunks)}]: {text_chunk}")
            
            phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_chunk}".strip())
            text_tokens, text_tokens_lens = text_collater([phone_tokens])

            enroll_x_lens = None
            if text_pr:
                text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
                text_prompts, enroll_x_lens = text_collater([text_prompts])
                text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
                text_tokens_lens += enroll_x_lens

            lang_final = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

            # Generate
            encoded_frames_chunk = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=-100,
                temperature=0.9,
                prompt_language=lang_pr,
                text_language=langs if accent == "no-accent" else lang_final,
                best_of=5,
            )

            frames = encoded_frames_chunk.permute(2,0,1)
            features = vocos.codes_to_features(frames)
            samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))
            all_audio_chunks.append(samples.squeeze(0).cpu().numpy())

        model.to('cpu')
        torch.cuda.empty_cache()

        # Concatenate chunks with crossfade
        if len(all_audio_chunks) == 1:
            audio_numpy = all_audio_chunks[0]
        else:
            # Simple concatenation with 50ms crossfade
            crossfade_samples = int(0.05 * 24000)  # 50ms
            audio_numpy = all_audio_chunks[0]
            
            for chunk in all_audio_chunks[1:]:
                if len(audio_numpy) < crossfade_samples or len(chunk) < crossfade_samples:
                    audio_numpy = np.concatenate([audio_numpy, chunk])
                else:
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    
                    audio_numpy[-crossfade_samples:] *= fade_out
                    chunk[:crossfade_samples] *= fade_in
                    audio_numpy[-crossfade_samples:] += chunk[:crossfade_samples]
                    audio_numpy = np.concatenate([audio_numpy, chunk[crossfade_samples:]])

        # Normalize final audio
        max_amp = np.abs(audio_numpy).max()
        if max_amp > 0:
            audio_numpy = audio_numpy * (0.95 / max_amp)

        message = f"✓ Generated {len(audio_numpy)/24000:.1f}s audio from {len(text_chunks)} chunk(s)"

        # Evaluate
        try:
            if temp_wav_path and os.path.isfile(temp_wav_path):
                evaluate_and_print_metrics(
                    audio_numpy, 24000, 
                    original_text=text, 
                    reference_prompt_wav_path=temp_wav_path, 
                    save_prefix="output"
                )
        except Exception as e:
            print(f"Evaluation failed: {e}")

        return message, (24000, audio_numpy)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        model.to('cpu')
        torch.cuda.empty_cache()
        return f"Error: {str(e)}", None

def main():
    try:
        load_speaker_diarization()
    except Exception as e:
        print(f"Failed to load speaker diarization: {e}")
        print("Continuing without speaker diarization...")
    
    app = gr.Blocks(title="VALL-E X")
    with app:
        gr.Markdown(top_md)
        # SIMPLIFIED UI - Only "Infer from audio" tab, removed unnecessary elements
        gr.Markdown(infer_from_audio_md)
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                      placeholder="Type your sentence here",
                                      value="Welcome back, Master. What can I do for you today?", 
                                      elem_id=f"tts-input")
                language_dropdown = gr.Dropdown(choices=['English'], value='English', label='language')
                accent_dropdown = gr.Dropdown(choices=['English'], value='English', label='accent')
                upload_audio_prompt = gr.Audio(label='Upload audio prompt', interactive=True)
                mic_audio_prompt = gr.Audio(source="microphone", label="Record with Microphone", interactive=True)
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(
                    infer_from_audio,
                    inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, mic_audio_prompt],
                    outputs=[text_output, audio_output]
                )

    try:
        # webbrowser.open("http://127.0.0.1:7860")
        app.launch(inbrowser=True)
    except Exception as e:
        print(f"Error launching app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
