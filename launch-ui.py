# import argparse
# import logging
# import os
# import pathlib
# import time
# import tempfile
# import platform
# import webbrowser
# import sys
# print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
# if(sys.version_info[0]<3 or sys.version_info[1]<7):
#     print("The Python version is too low and may cause problems")

# if platform.system().lower() == 'windows':
#     temp = pathlib.PosixPath
#     pathlib.PosixPath = pathlib.WindowsPath
# else:
#     temp = pathlib.WindowsPath
#     pathlib.WindowsPath = pathlib.PosixPath
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import langid
# langid.set_languages(['en'])
# import nltk
# nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

# import re  # Add regex import at module level

# import torch
# import torchaudio
# import random
# import soundfile as sf

# import numpy as np

# from data.tokenizer import (
#     AudioTokenizer,
#     tokenize_audio,
# )
# from data.collation import get_text_token_collater
# from models.vallex import VALLE
# from utils.g2p import PhonemeBpeTokenizer
# from descriptions import *
# from macros import *
# from examples import *

# import gradio as gr
# import whisper
# from vocos import Vocos
# import multiprocessing

# # ------------------------------
# # Simple KMeans Speaker Diarization
# # ------------------------------
# import librosa
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances

# import time
# import os
# import numpy as np

# # Lazy imports for embedding & ASR evaluation
# try:
#     import soundfile as sf
# except Exception:
#     sf = None
# try:
#     import torchaudio
# except Exception:
#     torchaudio = None

# # transformers Wav2Vec2 for speaker embeddings (lazy loaded)
# _w2v_processor = None
# _w2v_model = None
# def _ensure_w2v_loaded():
#     global _w2v_processor, _w2v_model
#     if _w2v_processor is None or _w2v_model is None:
#         try:
#             from transformers import Wav2Vec2Processor, Wav2Vec2Model
#             _w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model.eval()
#             for p in _w2v_model.parameters():
#                 p.requires_grad = False
#         except Exception as e:
#             _w2v_processor = None
#             _w2v_model = None
#             print(f"[Eval Helper] Failed to load Wav2Vec2 model: {e}")

# def get_embedding(wav_path):
#     import torch
#     global _w2v_processor, _w2v_model
#     if sf is None:
#         raise RuntimeError("soundfile (pysoundfile) is required for get_embedding")
#     _ensure_w2v_loaded()
#     if _w2v_processor is None or _w2v_model is None:
#         raise RuntimeError("Wav2Vec2 processor/model not loaded")
#     wav, sr = sf.read(wav_path)
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     wav = wav.astype("float32")
#     if sr != 16000:
#         if torchaudio is None:
#             raise RuntimeError("torchaudio required for resampling to 16k")
#         wav_t = torch.from_numpy(wav).unsqueeze(0)
#         wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
#         wav = wav_t.squeeze(0).numpy()
#         sr = 16000
#     inputs = _w2v_processor(wav, sampling_rate=sr, return_tensors="pt", padding=False)
#     input_values = inputs["input_values"]
#     if input_values.dim() == 3:
#         input_values = input_values.squeeze(1)
#     with torch.no_grad():
#         outputs = _w2v_model(input_values)
#     emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
#     return emb

# def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="eval"):
#     import torch
#     import torch.nn.functional as F
#     import jiwer
#     import whisper
#     from scipy.io.wavfile import write as write_wav
#     from sklearn.metrics.pairwise import cosine_similarity
    
#     audio = np.asarray(audio_numpy)
#     if audio.dtype not in (np.float32, np.float64):
#         peak = np.max(np.abs(audio)) if audio.size else 1.0
#         if peak > 0:
#             audio = audio.astype("float32") / float(peak)
#         else:
#             audio = audio.astype("float32")
#     else:
#         audio = audio.astype("float32")
#     peak = np.max(np.abs(audio)) if audio.size else 1.0
#     if peak > 1.0:
#         audio = audio / peak

#     timestamp = int(time.time())
#     out_filename = f"{save_prefix}_{timestamp}.wav"
#     try:
#         write_wav(out_filename, sample_rate, audio)
#     except Exception:
#         if sf is None:
#             print("[Eval] Could not save audio: scipy write failed and soundfile not present.")
#         else:
#             sf.write(out_filename, audio, sample_rate)
#     print(f"[Eval] Saved generated audio -> {out_filename}")

#     try:
#         try:
#             whisper_model
#             model_for_asr = whisper_model
#         except NameError:
#             model_for_asr = whisper.load_model("base")
#         tstart = time.time()
#         res = model_for_asr.transcribe(out_filename)
#         transcribed = res.get("text", "").strip()
#         tasr = time.time() - tstart
#     except Exception as e:
#         transcribed = "[ASR_ERROR]"
#         tasr = 0.0
#         print(f"[Eval] Whisper transcription failed: {e}")
    
#     wer_val = None
#     if original_text and original_text.strip() != "":
#         try:
#             # Clean text: remove special tokens and normalize
#             clean_original = re.sub(r'<\|[^|]+\|>', '', original_text)  # Remove language tokens
#             clean_original = clean_original.strip().lower()
#             clean_transcribed = transcribed.strip().lower()
            
#             # Remove multiple spaces and punctuation for better comparison
#             clean_original = re.sub(r'[^\w\s]', '', clean_original)
#             clean_transcribed = re.sub(r'[^\w\s]', '', clean_transcribed)
#             clean_original = re.sub(r'\s+', ' ', clean_original).strip()
#             clean_transcribed = re.sub(r'\s+', ' ', clean_transcribed).strip()
            
#             # Calculate WER only if both strings are non-empty
#             if clean_original and clean_transcribed:
#                 wer_val = jiwer.wer(clean_original, clean_transcribed)
            
#                 # Additional accuracy metric: simple word match
#                 orig_words = clean_original.split()
#                 trans_words = clean_transcribed.split()
#                 orig_words_set = set(orig_words)
#                 trans_words_set = set(trans_words)
                
#                 if len(orig_words) > 0:
#                     word_match = len(orig_words_set & trans_words_set) / len(orig_words_set)
#                     matched_words = len(orig_words_set & trans_words_set)
#                     total_words = len(orig_words_set)
#                     print(f"[Debug] Word Match Accuracy: {word_match*100:.1f}% ({matched_words}/{total_words} unique words)")
#                     print(f"[Debug] Original words: {orig_words}")
#                     print(f"[Debug] Transcribed words: {trans_words}")
#             else:
#                 print("[Debug] Skipping WER calculation - empty text after cleaning")
#         except Exception as e:
#             print(f"[Eval] WER computation error: {e}")
#             import traceback
#             traceback.print_exc()
#             wer_val = None

#     similarity = None
#     if reference_prompt_wav_path and os.path.isfile(reference_prompt_wav_path):
#         try:
#             print(f"[Eval] Computing speaker similarity using reference: {reference_prompt_wav_path}")
#             emb_gen = get_embedding(out_filename)
#             emb_ref = get_embedding(reference_prompt_wav_path)
#             emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
#             emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
#             similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
#             print(f"[Eval] Speaker similarity computed: {similarity:.4f}")
#         except Exception as e:
#             print(f"[Eval] Speaker similarity calculation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             similarity = None
#     else:
#         print(f"[Eval] No valid reference audio for speaker similarity (path: {reference_prompt_wav_path})")

#     print("\n=== ACCURACY METRICS ===")
#     if original_text:
#         print(f"Original Text     : {original_text}")
#     print(f"Transcribed Text  : {transcribed}")
#     if wer_val is not None:
#         print(f"WER               : {wer_val:.4f}")
#     else:
#         print("WER               : N/A")
#     print(f"ASR Time (sec)    : {tasr:.3f}")
#     if similarity is not None:
#         print(f"Speaker Similarity: {similarity:.4f} (cosine)")
#         if similarity >= 0.80:
#             remark = "Excellent cloning"
#         elif similarity >= 0.60:
#             remark = "Good cloning"
#         elif similarity >= 0.40:
#             remark = "Average cloning"
#         else:
#             remark = "Poor cloning"
#         print(f"Cloning Quality   : {remark}")
#     else:
#         print("Speaker Similarity: N/A")
#     print(f"Saved audio file  : {out_filename}")
#     print("=========================\n")

# def detect_speakers(audio_path, threshold=0.15):
#     try:
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         if duration < 1.0:
#             return 1, "✓ Single speaker (audio too short for analysis)"
        
#         energy = librosa.feature.rms(y=y)[0]
#         frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        
#         voiced_frames = []
#         for i in range(min(frames.shape[1], len(energy))):
#             if energy[i] > np.mean(energy) * 0.8:
#                 voiced_frames.append(frames[:, i])
        
#         if len(voiced_frames) < 5:
#             return 1, "✓ Single speaker (insufficient voice activity)"
        
#         embeddings = []
#         for frame in voiced_frames:
#             mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20)
#             emb = np.mean(mfcc, axis=1)
#             embeddings.append(emb)
        
#         if len(embeddings) < 10:
#             return 1, "✓ Single speaker (insufficient features)"
        
#         embeddings = np.array(embeddings)
        
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
#         labels = kmeans.labels_
        
#         cluster_0 = embeddings[labels == 0]
#         cluster_1 = embeddings[labels == 1]
        
#         if len(cluster_0) == 0 or len(cluster_1) == 0:
#             return 1, "✓ Single speaker detected"
        
#         centroid_0 = cluster_0.mean(axis=0)
#         centroid_1 = cluster_1.mean(axis=0)
        
#         distance = cosine_distances([centroid_0], [centroid_1])[0][0]
        
#         cluster_0_pct = len(cluster_0) / len(embeddings) * 100
#         cluster_1_pct = len(cluster_1) / len(embeddings) * 100
        
#         print(f"Cluster analysis: {cluster_0_pct:.1f}% vs {cluster_1_pct:.1f}%, distance={distance:.3f}")
        
#         if distance < threshold:
#             return 1, f"✓ Single speaker detected (distance={distance:.3f})"
#         else:
#             min_cluster_size = 15
#             if cluster_0_pct < min_cluster_size or cluster_1_pct < min_cluster_size:
#                 return 1, f"✓ Single speaker (minor variation detected, distance={distance:.3f})"
            
#             return 2, f"⚠ Multiple speakers detected ({cluster_0_pct:.0f}%/{cluster_1_pct:.0f}% split, distance={distance:.3f})"
    
#     except Exception as e:
#         print(f"Speaker detection error: {e}")
#         return 1, "✓ Single speaker (analysis failed, assuming single)"

# def check_audio_for_generation(audio_path):
#     num_speakers, detail_msg = detect_speakers(audio_path)
    
#     if num_speakers > 1:
#         error_msg = (
#             "❌ Multiple speakers detected in the audio!\n\n"
#             "This voice cloning system requires audio with only ONE speaker.\n"
#             f"Analysis: {detail_msg}\n\n"
#             "Please provide a different audio sample with only one person speaking."
#         )
#         return False, error_msg
#     else:
#         success_msg = f"✓ Audio validated for voice cloning\n{detail_msg}"
#         return True, success_msg

# thread_count = multiprocessing.cpu_count()
# print("Use",thread_count,"cpu cores for computing")

# torch.set_num_threads(thread_count)
# torch.set_num_interop_threads(thread_count)
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_set_profiling_mode(False)
# torch._C._set_graph_executor_optimize(False)

# text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
# text_collater = get_text_token_collater()

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda", 0)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")

# if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
# CHECKPOINT_PATH = "checkpoints/vallex-checkpoint.pt"

# os.makedirs("checkpoints", exist_ok=True)

# # Create prompts directory if it doesn't exist
# if not os.path.exists("./prompts/"): 
#     os.mkdir("./prompts/")

# if not os.path.isfile(CHECKPOINT_PATH):
#     import wget
#     try:
#         print("Model checkpoint not found. Downloading it now...")
#         logging.info("Downloading VALLE-X model (first time only)...")
#         wget.download(
#             "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
#             out=CHECKPOINT_PATH,
#             bar=wget.bar_adaptive
#         )
#         print("\nDownload complete!")
#     except Exception as e:
#         logging.info(e)
#         raise Exception(
#             "\nModel weights download failed.\n"
#             "Please manually download from https://huggingface.co/Plachta/VALL-E-X\n"
#             f"and put vallex-checkpoint.pt inside: {os.getcwd()}/checkpoints/"
#         )
# else:
#     print("✔ Using existing model checkpoint — skipping download.")

# model = VALLE(
#         N_DIM,
#         NUM_HEAD,
#         NUM_LAYERS,
#         norm_first=True,
#         add_prenet=False,
#         prefix_mode=PREFIX_MODE,
#         share_embedding=True,
#         nar_scale_factor=1.0,
#         prepend_bos=True,
#         num_quantizers=NUM_QUANTIZERS,
#     )
# checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu', weights_only=False)
# missing_keys, unexpected_keys = model.load_state_dict(
#     checkpoint["model"], strict=True
# )
# assert not missing_keys
# model.eval()

# audio_tokenizer = AudioTokenizer(device)
# vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
# try:
#     whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
# except Exception as e:
#     logging.info(e)
#     raise Exception(
#         "\n Whisper download failed or damaged, please go to "
#         "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
#         "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# preset_list = os.walk("./presets/").__next__()[2]
# preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

# speaker_diarization_model = None

# def load_speaker_diarization():
#     print("Speaker detection will be skipped")

# def check_single_speaker(audio_path):
#     try:
#         import librosa
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         if duration < 1.0:
#             return True, 1, ""
        
#         return True, 1, ""
        
#     except Exception as e:
#         print(f"Warning: Speaker diarization check failed: {e}")
#         return True, 1, ""

# def clear_prompts():
#     """Clean up old temporary files"""
#     try:
#         path = tempfile.gettempdir()
#         for eachfile in os.listdir(path):
#             filename = os.path.join(path, eachfile)
#             if os.path.isfile(filename) and filename.endswith(".npz"):
#                 lastmodifytime = os.stat(filename).st_mtime
#                 endfiletime = time.time() - 60
#                 if endfiletime > lastmodifytime:
#                     os.remove(filename)
#     except:
#         pass
    
#     # Also clean up temp enrollment audio files in current directory
#     try:
#         for eachfile in os.listdir(os.getcwd()):
#             if eachfile.startswith("temp_enroll_") and eachfile.endswith(".wav"):
#                 filename = os.path.join(os.getcwd(), eachfile)
#                 if os.path.isfile(filename):
#                     lastmodifytime = os.stat(filename).st_mtime
#                     endfiletime = time.time() - 300  # Delete files older than 5 minutes
#                     if endfiletime > lastmodifytime:
#                         os.remove(filename)
#                         print(f"[Cleanup] Deleted old temp file: {eachfile}")
#     except:
#         pass

# def transcribe_one(model, audio_path):
#     audio = whisper.load_audio(audio_path)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#     lang = max(probs, key=probs.get)
#     options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
#     result = whisper.decode(model, mel, options)
#     print(result.text)
#     text_pr = result.text
#     if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
#         text_pr += "."
#     return lang, text_pr

# def preprocess_text(text):
#     """
#     Preprocess text for better TTS handling of longer sentences.
#     - Normalize punctuation
#     - Add strategic pauses
#     - Handle sentence boundaries
#     """
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Ensure proper sentence endings
#     if text and text[-1] not in '.!?,;:':
#         text += '.'
    
#     # Add slight pauses after commas for better prosody
#     text = re.sub(r',(\S)', r', \1', text)
    
#     # Normalize multiple punctuation
#     text = re.sub(r'([.!?]){2,}', r'\1', text)
    
#     return text

# def make_prompt(name, wav, sr, save=True):
#     global whisper_model
#     whisper_model.to(device)
    
#     if not isinstance(wav, torch.FloatTensor):
#         wav = torch.tensor(wav)
    
#     if wav.abs().max() > 1:
#         wav = wav / wav.abs().max()
    
#     if wav.size(-1) == 2:
#         wav = wav.mean(-1, keepdim=False)
    
#     if wav.ndim == 1:
#         wav = wav.unsqueeze(0)
    
#     assert wav.ndim == 2 and wav.size(0) == 1, f"Expected shape (1, N), got {wav.shape}"
    
#     data = wav.squeeze(0).cpu().numpy()
#     data = np.clip(data, -1.0, 1.0)
#     data = data.astype(np.float32)
#     data = np.clip(data, -1.0, 1.0)
#     data_int16 = (data * 32767).astype(np.int16)
#     sf.write(f"./prompts/{name}.wav", data_int16, sr, subtype='PCM_16')
    
#     lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    
#     if lang != "en":
#         raise ValueError(f"Error: Only English audio is supported. Detected language: {lang}")
    
#     lang_token = lang2token[lang]
#     text = lang_token + text + lang_token
    
#     with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
#         f.write(text)
    
#     if not save:
#         os.remove(f"./prompts/{name}.wav")
#         os.remove(f"./prompts/{name}.txt")

#     whisper_model.cpu()
#     torch.cuda.empty_cache()
    
#     return text, lang

# @torch.no_grad()
# def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt):
#     global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
#     try:
#         # Clean up old temporary files first
#         clear_prompts()
        
#         # Preprocess the input text for better handling
#         text = preprocess_text(text)
#         print(f"[Debug] Preprocessed text: {text}")
        
#         audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt

#         if audio_prompt is None:
#             return "Error: No audio provided", None

#         temp_wav_path = None
#         sr = None
#         wav_pr = None

#         # Handle different Gradio audio input formats
#         if isinstance(audio_prompt, str):
#             # Format 1: Direct file path (string)
#             wav_pr, sr = torchaudio.load(audio_prompt)
#             temp_wav_path = audio_prompt
#         elif isinstance(audio_prompt, dict):
#             # Format 2: Dictionary with 'name' or 'path' key
#             audio_path = audio_prompt.get('name') or audio_prompt.get('path')
#             if audio_path and os.path.isfile(audio_path):
#                 wav_pr, sr = torchaudio.load(audio_path)
#                 temp_wav_path = audio_path
#             # Format 3: Dictionary with 'array' key (Gradio 4.x format)
#             elif 'array' in audio_prompt:
#                 arr = audio_prompt['array']
#                 if isinstance(arr, (list, tuple)) and len(arr) == 2:
#                     sr, data = arr
#                     wav_pr = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
#                 elif isinstance(arr, np.ndarray):
#                     # Sometimes it's just the array without sample rate
#                     wav_pr = torch.FloatTensor(arr)
#                     sr = audio_prompt.get('sample_rate', 24000)  # Default to 24000
#                 else:
#                     raise ValueError("Unsupported audio_prompt dict array format")
#             else:
#                 raise ValueError("Unsupported audio_prompt dict format; missing file path and array")
#         elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
#             # Format 4: Tuple (sample_rate, waveform) - older Gradio format
#             sr, wav_pr = audio_prompt
#             if not isinstance(wav_pr, torch.Tensor):
#                 wav_pr = torch.FloatTensor(wav_pr)
#         else:
#             return "Error: Unsupported audio format", None
        
#         # ALWAYS save temp wav for similarity calculation - THIS IS CRITICAL
#         needs_temp_file = temp_wav_path is None or not os.path.isfile(temp_wav_path)
#         if needs_temp_file:
#             tmp_path = os.path.join(os.getcwd(), f"temp_enroll_{int(time.time() * 1000)}.wav")
#             # Save properly normalized audio
#             wav_save = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
#             if wav_save.ndim > 1:
#                 wav_save = wav_save.mean(axis=0)  # Convert to mono if stereo
#             wav_save = np.clip(wav_save, -1.0, 1.0)
#             wav_save_int16 = (wav_save * 32767).astype(np.int16)
#             sf.write(tmp_path, wav_save_int16, sr, subtype='PCM_16')
#             temp_wav_path = tmp_path
#             print(f"[Debug] Created temp reference audio: {temp_wav_path}")
        
#         # Verify the temp file exists
#         if not os.path.isfile(temp_wav_path):
#             print(f"[ERROR] Reference audio file does not exist: {temp_wav_path}")
#             return "Error: Could not create reference audio file", None
        
#         print(f"[Debug] Using reference audio path: {temp_wav_path} (exists: {os.path.isfile(temp_wav_path)})")

#         if temp_wav_path:
#             is_valid, check_message = check_audio_for_generation(temp_wav_path)
#             if not is_valid:
#                 return check_message, None
#             else:
#                 print(check_message)

#         if not isinstance(wav_pr, torch.Tensor):
#             wav_pr = torch.FloatTensor(wav_pr)

#         if wav_pr.abs().max() > 1:
#             wav_pr = wav_pr / wav_pr.abs().max()

#         if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
#             wav_pr = wav_pr.mean(dim=0, keepdim=True)
#         elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
#             wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)

#         if wav_pr.ndim == 1:
#             wav_pr = wav_pr.unsqueeze(0)

#         assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"

#         text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)

#         if language == 'auto-detect':
#             lang_token = lang2token[langid.classify(text)[0]]
#         else:
#             lang_token = langdropdown2token[language]
#         lang = token2lang[lang_token]
#         text = lang_token + text + lang_token

#         model.to(device)

#         encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
#         audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

#         logging.info(f"synthesize text: {text}")
#         phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
#         text_tokens, text_tokens_lens = text_collater([phone_tokens])

#         enroll_x_lens = None
#         if text_pr:
#             text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
#             text_prompts, enroll_x_lens = text_collater([text_prompts])

#         text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#         text_tokens_lens += enroll_x_lens
#         lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

#         gen_start = time.time()
#         # IMPROVED GENERATION PARAMETERS for longer text
#         encoded_frames = model.inference(
#             text_tokens.to(device),
#             text_tokens_lens.to(device),
#             audio_prompts,
#             enroll_x_lens=enroll_x_lens,
#             top_k=-100,           # Balanced sampling
#             temperature=0.75,     # Lower temperature for more stable longer outputs
#             prompt_language=lang_pr,
#             text_language=langs if accent == "no-accent" else lang,
#             best_of=3,            # Balance quality vs speed for longer text
#         )

# def main():
#     app = gr.Blocks(title="VALL-E X")
#     with app:
#         gr.Markdown(top_md)
#         # SIMPLIFIED UI - Only "Infer from audio" tab, removed unnecessary elements
#         gr.Markdown(infer_from_audio_md)
#         with gr.Row():
#             with gr.Column():
#                 textbox = gr.TextArea(label="Text",
#                                       placeholder="Type your sentence here",
#                                       value="Welcome back, Master. What can I do for you today?", 
#                                       elem_id=f"tts-input")
#                 language_dropdown = gr.Dropdown(choices=['English'], value='English', label='language')
#                 accent_dropdown = gr.Dropdown(choices=['English'], value='English', label='accent')
#                 upload_audio_prompt = gr.Audio(label='Upload audio prompt', interactive=True)
#                 mic_audio_prompt = gr.Audio(source="microphone", label="Record with Microphone", interactive=True)
#             with gr.Column():
#                 text_output = gr.Textbox(label="Message")
#                 audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                 btn = gr.Button("Generate!")
#                 btn.click(
#                     infer_from_audio,
#                     inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, mic_audio_prompt],
#                     outputs=[text_output, audio_output]
#                 )

#     try:
#         webbrowser.open("http://127.0.0.1:7860")
#         app.launch(inbrowser=True)
#     except Exception as e:
#         print(f"Error launching app: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     formatter = (
#         "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
#     )
#     logging.basicConfig(format=formatter, level=logging.INFO)
    
#     try:
#         main()
#     except Exception as e:
#         print(f"Fatal error: {e}")
#         import traceback
#         traceback.print_exc()











# good but little fat
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

def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="eval"):
    import torch
    import torch.nn.functional as F
    import jiwer
    import whisper
    from scipy.io.wavfile import write as write_wav
    from sklearn.metrics.pairwise import cosine_similarity
    
    audio = np.asarray(audio_numpy)
    if audio.dtype not in (np.float32, np.float64):
        peak = np.max(np.abs(audio)) if audio.size else 1.0
        if peak > 0:
            audio = audio.astype("float32") / float(peak)
        else:
            audio = audio.astype("float32")
    else:
        audio = audio.astype("float32")
    peak = np.max(np.abs(audio)) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak

    timestamp = int(time.time())
    out_filename = f"{save_prefix}_{timestamp}.wav"
    try:
        write_wav(out_filename, sample_rate, audio)
    except Exception:
        if sf is None:
            print("[Eval] Could not save audio: scipy write failed and soundfile not present.")
        else:
            sf.write(out_filename, audio, sample_rate)
    print(f"[Eval] Saved generated audio -> {out_filename}")

    try:
        try:
            whisper_model
            model_for_asr = whisper_model
        except NameError:
            model_for_asr = whisper.load_model("base")
        tstart = time.time()
        res = model_for_asr.transcribe(out_filename)
        transcribed = res.get("text", "").strip()
        tasr = time.time() - tstart
    except Exception as e:
        transcribed = "[ASR_ERROR]"
        tasr = 0.0
        print(f"[Eval] Whisper transcription failed: {e}")
    
    wer_val = None
    if original_text and original_text.strip() != "":
        try:
            # Clean text: remove special tokens and normalize
            clean_original = re.sub(r'<\|[^|]+\|>', '', original_text)  # Remove language tokens
            clean_original = clean_original.strip().lower()
            clean_transcribed = transcribed.strip().lower()
            
            # Remove multiple spaces and punctuation for better comparison
            clean_original = re.sub(r'[^\w\s]', '', clean_original)
            clean_transcribed = re.sub(r'[^\w\s]', '', clean_transcribed)
            clean_original = re.sub(r'\s+', ' ', clean_original).strip()
            clean_transcribed = re.sub(r'\s+', ' ', clean_transcribed).strip()
            
            # Calculate WER only if both strings are non-empty
            if clean_original and clean_transcribed:
                wer_val = jiwer.wer(clean_original, clean_transcribed)
            
                # Additional accuracy metric: simple word match
                orig_words = clean_original.split()
                trans_words = clean_transcribed.split()
                orig_words_set = set(orig_words)
                trans_words_set = set(trans_words)
                
                if len(orig_words) > 0:
                    word_match = len(orig_words_set & trans_words_set) / len(orig_words_set)
                    matched_words = len(orig_words_set & trans_words_set)
                    total_words = len(orig_words_set)
                    print(f"[Debug] Word Match Accuracy: {word_match*100:.1f}% ({matched_words}/{total_words} unique words)")
                    print(f"[Debug] Original words: {orig_words}")
                    print(f"[Debug] Transcribed words: {trans_words}")
            else:
                print("[Debug] Skipping WER calculation - empty text after cleaning")
        except Exception as e:
            print(f"[Eval] WER computation error: {e}")
            import traceback
            traceback.print_exc()
            wer_val = None

    similarity = None
    if reference_prompt_wav_path:
        try:
            emb_gen = get_embedding(out_filename)
            emb_ref = get_embedding(reference_prompt_wav_path)
            emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
            emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
            similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
        except Exception as e:
            print(f"[Eval] Speaker similarity calculation failed: {e}")
            similarity = None

    print("\n=== ACCURACY METRICS ===")
    if original_text:
        print(f"Original Text     : {original_text}")
    print(f"Transcribed Text  : {transcribed}")
    if wer_val is not None:
        print(f"WER               : {wer_val:.4f}")
    else:
        print("WER               : N/A")
    print(f"ASR Time (sec)    : {tasr:.3f}")
    if similarity is not None:
        print(f"Speaker Similarity: {similarity:.4f} (cosine)")
        if similarity >= 0.80:
            remark = "Excellent cloning"
        elif similarity >= 0.60:
            remark = "Good cloning"
        elif similarity >= 0.40:
            remark = "Average cloning"
        else:
            remark = "Poor cloning"
        print(f"Cloning Quality   : {remark}")
    else:
        print("Speaker Similarity: N/A")
    print(f"Saved audio file  : {out_filename}")
    print("=========================\n")

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
            f"Analysis: {detail_msg}\n\n"
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

@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt):
    global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
    try:
        # Clean up old temporary files first
        clear_prompts()
        
        audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt

        if audio_prompt is None:
            return "Error: No audio provided", None

        temp_wav_path = None
        sr = None
        wav_pr = None

        # Handle different Gradio audio input formats
        if isinstance(audio_prompt, str):
            # Format 1: Direct file path (string)
            wav_pr, sr = torchaudio.load(audio_prompt)
            temp_wav_path = audio_prompt
        elif isinstance(audio_prompt, dict):
            # Format 2: Dictionary with 'name' or 'path' key
            audio_path = audio_prompt.get('name') or audio_prompt.get('path')
            if audio_path and os.path.isfile(audio_path):
                wav_pr, sr = torchaudio.load(audio_path)
                temp_wav_path = audio_path
            # Format 3: Dictionary with 'array' key (Gradio 4.x format)
            elif 'array' in audio_prompt:
                arr = audio_prompt['array']
                if isinstance(arr, (list, tuple)) and len(arr) == 2:
                    sr, data = arr
                    wav_pr = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
                elif isinstance(arr, np.ndarray):
                    # Sometimes it's just the array without sample rate
                    wav_pr = torch.FloatTensor(arr)
                    sr = audio_prompt.get('sample_rate', 24000)  # Default to 24000
                else:
                    raise ValueError("Unsupported audio_prompt dict array format")
            else:
                raise ValueError("Unsupported audio_prompt dict format; missing file path and array")
        elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
            # Format 4: Tuple (sample_rate, waveform) - older Gradio format
            sr, wav_pr = audio_prompt
            if not isinstance(wav_pr, torch.Tensor):
                wav_pr = torch.FloatTensor(wav_pr)
        else:
            return "Error: Unsupported audio format", None
        
        # ALWAYS save temp wav for similarity calculation
        if temp_wav_path is None or not os.path.isfile(temp_wav_path):
            tmp_path = os.path.join(os.getcwd(), f"temp_enroll_{int(time.time())}.wav")
            # Save properly normalized audio
            wav_save = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
            wav_save = np.clip(wav_save, -1.0, 1.0)
            wav_save_int16 = (wav_save * 32767).astype(np.int16)
            sf.write(tmp_path, wav_save_int16, sr, subtype='PCM_16')
            temp_wav_path = tmp_path
            print(f"[Debug] Saved temp reference audio: {temp_wav_path}")

        if temp_wav_path:
            is_valid, check_message = check_audio_for_generation(temp_wav_path)
            if not is_valid:
                return check_message, None
            else:
                print(check_message)

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

        assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"

        text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)

        if language == 'auto-detect':
            lang_token = lang2token[langid.classify(text)[0]]
        else:
            lang_token = langdropdown2token[language]
        lang = token2lang[lang_token]
        text = lang_token + text + lang_token

        model.to(device)

        encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
        audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

        logging.info(f"synthesize text: {text}")
        phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
        text_tokens, text_tokens_lens = text_collater([phone_tokens])

        enroll_x_lens = None
        if text_pr:
            text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
            text_prompts, enroll_x_lens = text_collater([text_prompts])

        text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
        text_tokens_lens += enroll_x_lens
        lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

        gen_start = time.time()
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,          # More variety in generation
            temperature=0.85,     # More stable output
            prompt_language=lang_pr,
            text_language=langs if accent == "no-accent" else lang,
            best_of=5,           # Better quality selection
        )
        gen_end = time.time()
        
        print(f"[Debug] Generation took {gen_end - gen_start:.2f} seconds")

        frames = encoded_frames.permute(2,0,1)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        torch.cuda.empty_cache()

        audio_numpy = samples.squeeze(0).cpu().numpy()
        
        # Post-process audio to reduce silence/breaks
        # Normalize audio amplitude
        max_amp = np.abs(audio_numpy).max()
        if max_amp > 0:
            audio_numpy = audio_numpy * (0.95 / max_amp)  # Normalize to 95% of max
        
        if len(audio_numpy) > 24000 * 300:
            logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
            audio_numpy = audio_numpy[:24000 * 300]
        
        print(f"[Debug] Generated audio duration: {len(audio_numpy)/24000:.2f}s, amplitude range: [{audio_numpy.min():.3f}, {audio_numpy.max():.3f}]")

        message = f"✓ Generated successfully!\nPrompt: {text_pr}\nSynthesized: {text}\nDuration: {len(audio_numpy)/24000:.1f}s"

        try:
            # Make sure reference path exists before evaluation
            if temp_wav_path and os.path.isfile(temp_wav_path):
                print(f"[Debug] Using reference audio: {temp_wav_path}")
                evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=temp_wav_path, save_prefix="infer_from_audio")
            else:
                print(f"[Warning] Reference audio not found: {temp_wav_path}")
                evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=None, save_prefix="infer_from_audio")
        except Exception as e:
            print(f"[Eval] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary enrollment file after a delay
            if temp_wav_path and temp_wav_path.startswith(os.path.join(os.getcwd(), "temp_enroll_")):
                try:
                    # Wait a bit to ensure evaluation is complete
                    import threading
                    def delayed_cleanup(path):
                        time.sleep(60)  # Wait 1 minute
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                                print(f"[Cleanup] Removed temp file: {path}")
                        except:
                            pass
                    cleanup_thread = threading.Thread(target=delayed_cleanup, args=(temp_wav_path,))
                    cleanup_thread.daemon = True
                    cleanup_thread.start()
                except:
                    pass

        return message, (24000, audio_numpy)

    except Exception as e:
        logging.error(f"Error in infer_from_audio: {str(e)}")
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
        webbrowser.open("http://127.0.0.1:7860")
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






# import argparse
# import logging
# import os
# import pathlib
# import time
# import tempfile
# import platform
# import webbrowser
# import sys
# print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
# if(sys.version_info[0]<3 or sys.version_info[1]<7):
#     print("The Python version is too low and may cause problems")

# if platform.system().lower() == 'windows':
#     temp = pathlib.PosixPath
#     pathlib.PosixPath = pathlib.WindowsPath
# else:
#     temp = pathlib.WindowsPath
#     pathlib.WindowsPath = pathlib.PosixPath
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import langid
# langid.set_languages(['en'])
# import nltk
# nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

# import re

# import torch
# import torchaudio
# import random
# import soundfile as sf
# import numpy as np

# from data.tokenizer import (
#     AudioTokenizer,
#     tokenize_audio,
# )
# from data.collation import get_text_token_collater
# from models.vallex import VALLE
# from utils.g2p import PhonemeBpeTokenizer
# from descriptions import *
# from macros import *
# from examples import *

# import gradio as gr
# import whisper
# from vocos import Vocos
# import multiprocessing

# # ------------------------------
# # Simple KMeans Speaker Diarization
# # ------------------------------
# import librosa
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances

# # transformers Wav2Vec2 for speaker embeddings (lazy loaded)
# _w2v_processor = None
# _w2v_model = None
# def _ensure_w2v_loaded():
#     global _w2v_processor, _w2v_model
#     if _w2v_processor is None or _w2v_model is None:
#         try:
#             from transformers import Wav2Vec2Processor, Wav2Vec2Model
#             _w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model.eval()
#             for p in _w2v_model.parameters():
#                 p.requires_grad = False
#             print("[Debug] Wav2Vec2 model loaded successfully for similarity calculation")
#         except Exception as e:
#             _w2v_processor = None
#             _w2v_model = None
#             print(f"[Eval Helper] Failed to load Wav2Vec2 model: {e}")

# def get_embedding(wav_path):
#     import torch
#     global _w2v_processor, _w2v_model
#     if sf is None:
#         raise RuntimeError("soundfile (pysoundfile) is required for get_embedding")
    
#     print(f"[Debug] Getting embedding for: {wav_path}")
#     print(f"[Debug] File exists: {os.path.exists(wav_path)}")
    
#     _ensure_w2v_loaded()
#     if _w2v_processor is None or _w2v_model is None:
#         raise RuntimeError("Wav2Vec2 processor/model not loaded")
    
#     wav, sr = sf.read(wav_path)
#     print(f"[Debug] Loaded audio: shape={wav.shape}, sr={sr}")
    
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     wav = wav.astype("float32")
    
#     if sr != 16000:
#         if torchaudio is None:
#             raise RuntimeError("torchaudio required for resampling to 16k")
#         wav_t = torch.from_numpy(wav).unsqueeze(0)
#         wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
#         wav = wav_t.squeeze(0).numpy()
#         sr = 16000
    
#     inputs = _w2v_processor(wav, sampling_rate=sr, return_tensors="pt", padding=False)
#     input_values = inputs["input_values"]
#     if input_values.dim() == 3:
#         input_values = input_values.squeeze(1)
    
#     with torch.no_grad():
#         outputs = _w2v_model(input_values)
    
#     emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
#     print(f"[Debug] Embedding computed: shape={emb.shape}")
#     return emb

# def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="eval"):
#     import torch
#     import torch.nn.functional as F
#     import jiwer
#     import whisper
#     from scipy.io.wavfile import write as write_wav
#     from sklearn.metrics.pairwise import cosine_similarity
    
#     audio = np.asarray(audio_numpy)
#     if audio.dtype not in (np.float32, np.float64):
#         peak = np.max(np.abs(audio)) if audio.size else 1.0
#         if peak > 0:
#             audio = audio.astype("float32") / float(peak)
#         else:
#             audio = audio.astype("float32")
#     else:
#         audio = audio.astype("float32")
#     peak = np.max(np.abs(audio)) if audio.size else 1.0
#     if peak > 1.0:
#         audio = audio / peak

#     timestamp = int(time.time())
#     out_filename = f"{save_prefix}_{timestamp}.wav"
    
#     # Save with proper format
#     try:
#         audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
#         sf.write(out_filename, audio_int16, sample_rate, subtype='PCM_16')
#         print(f"[Eval] Saved generated audio -> {out_filename}")
#     except Exception as e:
#         print(f"[Eval] Failed to save audio: {e}")
#         return

#     # Whisper transcription
#     try:
#         try:
#             whisper_model
#             model_for_asr = whisper_model
#         except NameError:
#             model_for_asr = whisper.load_model("base")
#         tstart = time.time()
#         res = model_for_asr.transcribe(out_filename)
#         transcribed = res.get("text", "").strip()
#         tasr = time.time() - tstart
#     except Exception as e:
#         transcribed = "[ASR_ERROR]"
#         tasr = 0.0
#         print(f"[Eval] Whisper transcription failed: {e}")
    
#     # WER calculation
#     wer_val = None
#     if original_text and original_text.strip() != "":
#         try:
#             clean_original = re.sub(r'<\|[^|]+\|>', '', original_text)
#             clean_original = clean_original.strip().lower()
#             clean_transcribed = transcribed.strip().lower()
            
#             clean_original = re.sub(r'[^\w\s]', '', clean_original)
#             clean_transcribed = re.sub(r'[^\w\s]', '', clean_transcribed)
#             clean_original = re.sub(r'\s+', ' ', clean_original).strip()
#             clean_transcribed = re.sub(r'\s+', ' ', clean_transcribed).strip()
            
#             if clean_original and clean_transcribed:
#                 wer_val = jiwer.wer(clean_original, clean_transcribed)
            
#                 orig_words = clean_original.split()
#                 trans_words = clean_transcribed.split()
#                 orig_words_set = set(orig_words)
#                 trans_words_set = set(trans_words)
                
#                 if len(orig_words) > 0:
#                     word_match = len(orig_words_set & trans_words_set) / len(orig_words_set)
#                     matched_words = len(orig_words_set & trans_words_set)
#                     total_words = len(orig_words_set)
#                     print(f"[Debug] Word Match Accuracy: {word_match*100:.1f}% ({matched_words}/{total_words} unique words)")
#         except Exception as e:
#             print(f"[Eval] WER computation error: {e}")
#             wer_val = None

#     # Similarity calculation with better error handling
#     similarity = None
#     if reference_prompt_wav_path:
#         print(f"[Debug] Starting similarity calculation...")
#         print(f"[Debug] Reference path: {reference_prompt_wav_path}")
#         print(f"[Debug] Generated path: {out_filename}")
#         print(f"[Debug] Reference exists: {os.path.exists(reference_prompt_wav_path)}")
#         print(f"[Debug] Generated exists: {os.path.exists(out_filename)}")
        
#         try:
#             # Make sure both files exist and are valid
#             if not os.path.exists(reference_prompt_wav_path):
#                 print(f"[Error] Reference audio file not found: {reference_prompt_wav_path}")
#             elif not os.path.exists(out_filename):
#                 print(f"[Error] Generated audio file not found: {out_filename}")
#             else:
#                 # Check file sizes
#                 ref_size = os.path.getsize(reference_prompt_wav_path)
#                 gen_size = os.path.getsize(out_filename)
#                 print(f"[Debug] Reference file size: {ref_size} bytes")
#                 print(f"[Debug] Generated file size: {gen_size} bytes")
                
#                 if ref_size < 1000 or gen_size < 1000:
#                     print(f"[Warning] Audio files too small, similarity may be unreliable")
                
#                 emb_gen = get_embedding(out_filename)
#                 emb_ref = get_embedding(reference_prompt_wav_path)
                
#                 emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
#                 emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
                
#                 similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
#                 print(f"[Debug] Similarity calculated: {similarity:.4f}")
#         except Exception as e:
#             print(f"[Eval] Speaker similarity calculation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             similarity = None
#     else:
#         print(f"[Debug] No reference audio path provided for similarity calculation")

#     # Print results
#     print("\n=== ACCURACY METRICS ===")
#     if original_text:
#         print(f"Original Text     : {original_text}")
#     print(f"Transcribed Text  : {transcribed}")
#     if wer_val is not None:
#         print(f"WER               : {wer_val:.4f}")
#     else:
#         print("WER               : N/A")
#     print(f"ASR Time (sec)    : {tasr:.3f}")
#     if similarity is not None:
#         print(f"Speaker Similarity: {similarity:.4f} (cosine)")
#         if similarity >= 0.80:
#             remark = "Excellent cloning"
#         elif similarity >= 0.60:
#             remark = "Good cloning"
#         elif similarity >= 0.40:
#             remark = "Average cloning"
#         else:
#             remark = "Poor cloning"
#         print(f"Cloning Quality   : {remark}")
#     else:
#         print("Speaker Similarity: N/A (check logs above for errors)")
#     print(f"Saved audio file  : {out_filename}")
#     print("=========================\n")

# def detect_speakers(audio_path, threshold=0.15):
#     try:
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         if duration < 1.0:
#             return 1, "✓ Single speaker (audio too short for analysis)"
        
#         energy = librosa.feature.rms(y=y)[0]
#         frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        
#         voiced_frames = []
#         for i in range(min(frames.shape[1], len(energy))):
#             if energy[i] > np.mean(energy) * 0.8:
#                 voiced_frames.append(frames[:, i])
        
#         if len(voiced_frames) < 5:
#             return 1, "✓ Single speaker (insufficient voice activity)"
        
#         embeddings = []
#         for frame in voiced_frames:
#             mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20)
#             emb = np.mean(mfcc, axis=1)
#             embeddings.append(emb)
        
#         if len(embeddings) < 10:
#             return 1, "✓ Single speaker (insufficient features)"
        
#         embeddings = np.array(embeddings)
        
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
#         labels = kmeans.labels_
        
#         cluster_0 = embeddings[labels == 0]
#         cluster_1 = embeddings[labels == 1]
        
#         if len(cluster_0) == 0 or len(cluster_1) == 0:
#             return 1, "✓ Single speaker detected"
        
#         centroid_0 = cluster_0.mean(axis=0)
#         centroid_1 = cluster_1.mean(axis=0)
        
#         distance = cosine_distances([centroid_0], [centroid_1])[0][0]
        
#         cluster_0_pct = len(cluster_0) / len(embeddings) * 100
#         cluster_1_pct = len(cluster_1) / len(embeddings) * 100
        
#         print(f"Cluster analysis: {cluster_0_pct:.1f}% vs {cluster_1_pct:.1f}%, distance={distance:.3f}")
        
#         if distance < threshold:
#             return 1, f"✓ Single speaker detected (distance={distance:.3f})"
#         else:
#             min_cluster_size = 15
#             if cluster_0_pct < min_cluster_size or cluster_1_pct < min_cluster_size:
#                 return 1, f"✓ Single speaker (minor variation detected, distance={distance:.3f})"
            
#             return 2, f"⚠ Multiple speakers detected ({cluster_0_pct:.0f}%/{cluster_1_pct:.0f}% split, distance={distance:.3f})"
    
#     except Exception as e:
#         print(f"Speaker detection error: {e}")
#         return 1, "✓ Single speaker (analysis failed, assuming single)"

# def check_audio_for_generation(audio_path):
#     num_speakers, detail_msg = detect_speakers(audio_path)
    
#     if num_speakers > 1:
#         error_msg = (
#             "❌ Multiple speakers detected in the audio!\n\n"
#             "This voice cloning system requires audio with only ONE speaker.\n"
#             f"Analysis: {detail_msg}\n\n"
#             "Please provide a different audio sample with only one person speaking."
#         )
#         return False, error_msg
#     else:
#         success_msg = f"✓ Audio validated for voice cloning\n{detail_msg}"
#         return True, success_msg

# thread_count = multiprocessing.cpu_count()
# print("Use",thread_count,"cpu cores for computing")

# torch.set_num_threads(thread_count)
# torch.set_num_interop_threads(thread_count)
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_set_profiling_mode(False)
# torch._C._set_graph_executor_optimize(False)

# text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
# text_collater = get_text_token_collater()

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda", 0)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")

# if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
# if not os.path.exists("./prompts/"): os.mkdir("./prompts/")
# CHECKPOINT_PATH = "checkpoints/vallex-checkpoint.pt"

# os.makedirs("checkpoints", exist_ok=True)
# os.makedirs("prompts", exist_ok=True)

# if not os.path.isfile(CHECKPOINT_PATH):
#     import wget
#     try:
#         print("Model checkpoint not found. Downloading it now...")
#         logging.info("Downloading VALLE-X model (first time only)...")
#         wget.download(
#             "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
#             out=CHECKPOINT_PATH,
#             bar=wget.bar_adaptive
#         )
#         print("\nDownload complete!")
#     except Exception as e:
#         logging.info(e)
#         raise Exception(
#             "\nModel weights download failed.\n"
#             "Please manually download from https://huggingface.co/Plachta/VALL-E-X\n"
#             f"and put vallex-checkpoint.pt inside: {os.getcwd()}/checkpoints/"
#         )
# else:
#     print("✔ Using existing model checkpoint — skipping download.")

# model = VALLE(
#         N_DIM,
#         NUM_HEAD,
#         NUM_LAYERS,
#         norm_first=True,
#         add_prenet=False,
#         prefix_mode=PREFIX_MODE,
#         share_embedding=True,
#         nar_scale_factor=1.0,
#         prepend_bos=True,
#         num_quantizers=NUM_QUANTIZERS,
#     )
# checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu', weights_only=False)
# missing_keys, unexpected_keys = model.load_state_dict(
#     checkpoint["model"], strict=True
# )
# assert not missing_keys
# model.eval()

# audio_tokenizer = AudioTokenizer(device)
# vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
# try:
#     whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
# except Exception as e:
#     logging.info(e)
#     raise Exception(
#         "\n Whisper download failed or damaged, please go to "
#         "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
#         "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# preset_list = os.walk("./presets/").__next__()[2]
# preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

# speaker_diarization_model = None

# def load_speaker_diarization():
#     print("Speaker detection will be skipped")

# def check_single_speaker(audio_path):
#     try:
#         import librosa
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         if duration < 1.0:
#             return True, 1, ""
        
#         return True, 1, ""
        
#     except Exception as e:
#         print(f"Warning: Speaker diarization check failed: {e}")
#         return True, 1, ""

# def clear_prompts():
#     """Clean up old temporary files"""
#     try:
#         path = tempfile.gettempdir()
#         for eachfile in os.listdir(path):
#             filename = os.path.join(path, eachfile)
#             if os.path.isfile(filename) and filename.endswith(".npz"):
#                 lastmodifytime = os.stat(filename).st_mtime
#                 endfiletime = time.time() - 60
#                 if endfiletime > lastmodifytime:
#                     os.remove(filename)
#     except:
#         pass
    
#     # Clean up temp enrollment audio files
#     try:
#         for eachfile in os.listdir(os.getcwd()):
#             if eachfile.startswith("temp_enroll_") and eachfile.endswith(".wav"):
#                 filename = os.path.join(os.getcwd(), eachfile)
#                 if os.path.isfile(filename):
#                     lastmodifytime = os.stat(filename).st_mtime
#                     endfiletime = time.time() - 300
#                     if endfiletime > lastmodifytime:
#                         os.remove(filename)
#                         print(f"[Cleanup] Deleted old temp file: {eachfile}")
#     except:
#         pass

# def transcribe_one(model, audio_path):
#     audio = whisper.load_audio(audio_path)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#     lang = max(probs, key=probs.get)
#     options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
#     result = whisper.decode(model, mel, options)
#     print(result.text)
#     text_pr = result.text
#     if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
#         text_pr += "."
#     return lang, text_pr

# def make_prompt(name, wav, sr, save=True):
#     global whisper_model
#     whisper_model.to(device)
    
#     if not isinstance(wav, torch.FloatTensor):
#         wav = torch.tensor(wav)
    
#     if wav.abs().max() > 1:
#         wav = wav / wav.abs().max()
    
#     if wav.size(-1) == 2:
#         wav = wav.mean(-1, keepdim=False)
    
#     if wav.ndim == 1:
#         wav = wav.unsqueeze(0)
    
#     assert wav.ndim == 2 and wav.size(0) == 1, f"Expected shape (1, N), got {wav.shape}"
    
#     data = wav.squeeze(0).cpu().numpy()
#     data = np.clip(data, -1.0, 1.0)
#     data = data.astype(np.float32)
#     data = np.clip(data, -1.0, 1.0)
#     data_int16 = (data * 32767).astype(np.int16)
#     sf.write(f"./prompts/{name}.wav", data_int16, sr, subtype='PCM_16')
    
#     lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    
#     if lang != "en":
#         raise ValueError(f"Error: Only English audio is supported. Detected language: {lang}")
    
#     lang_token = lang2token[lang]
#     text = lang_token + text + lang_token
    
#     with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
#         f.write(text)
    
#     if not save:
#         os.remove(f"./prompts/{name}.wav")
#         os.remove(f"./prompts/{name}.txt")

#     whisper_model.cpu()
#     torch.cuda.empty_cache()
    
#     return text, lang

# @torch.no_grad()
# def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt):
#     global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
#     try:
#         clear_prompts()
        
#         # Validate and clean input text
#         if not text or not text.strip():
#             return "Error: Please enter some text to synthesize", None
        
#         # Clean the input text - remove problematic characters
#         text = text.strip()
#         # Remove multiple newlines and replace with space
#         text = re.sub(r'\n+', ' ', text)
#         # Remove multiple spaces
#         text = re.sub(r'\s+', ' ', text)
#         # Remove any control characters or special Unicode
#         text = ''.join(char for char in text if char.isprintable() or char.isspace())
#         # Keep only alphanumeric, spaces, and basic punctuation
#         text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text, flags=re.UNICODE)
#         text = text.strip()
        
#         # Validate text again after cleaning
#         if not text:
#             return "Error: Text is empty after cleaning. Please enter valid text with words and letters.", None
        
#         # Check if text contains at least some alphabetic characters
#         if not any(c.isalpha() for c in text):
#             return "Error: Text must contain at least some letters or words. Numbers and symbols alone cannot be synthesized.", None
        
#         print(f"[Debug] Cleaned input text: {text}")
        
#         audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt

#         if audio_prompt is None:
#             return "Error: No audio provided", None

#         temp_wav_path = None
#         sr = None
#         wav_pr = None

#         # Handle different Gradio audio formats
#         if isinstance(audio_prompt, str):
#             wav_pr, sr = torchaudio.load(audio_prompt)
#             temp_wav_path = audio_prompt
#         elif isinstance(audio_prompt, dict):
#             audio_path = audio_prompt.get('name') or audio_prompt.get('path')
#             if audio_path and os.path.isfile(audio_path):
#                 wav_pr, sr = torchaudio.load(audio_path)
#                 temp_wav_path = audio_path
#             elif 'array' in audio_prompt:
#                 arr = audio_prompt['array']
#                 if isinstance(arr, (list, tuple)) and len(arr) == 2:
#                     sr, data = arr
#                     wav_pr = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
#                 elif isinstance(arr, np.ndarray):
#                     wav_pr = torch.FloatTensor(arr)
#                     sr = audio_prompt.get('sample_rate', 24000)
#                 else:
#                     raise ValueError("Unsupported audio_prompt dict array format")
#             else:
#                 raise ValueError("Unsupported audio_prompt dict format")
#         elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
#             sr, wav_pr = audio_prompt
#             if not isinstance(wav_pr, torch.Tensor):
#                 wav_pr = torch.FloatTensor(wav_pr)
#         else:
#             return "Error: Unsupported audio format", None
        
#         # ALWAYS save temp wav for similarity
#         if temp_wav_path is None or not os.path.isfile(temp_wav_path):
#             tmp_path = os.path.join(os.getcwd(), f"temp_enroll_{int(time.time()*1000)}.wav")
#             wav_save = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
#             wav_save = np.clip(wav_save, -1.0, 1.0)
#             wav_save_int16 = (wav_save * 32767).astype(np.int16)
#             sf.write(tmp_path, wav_save_int16, sr, subtype='PCM_16')
#             temp_wav_path = tmp_path
#             print(f"[Debug] Saved temp reference audio: {temp_wav_path}")
#             print(f"[Debug] File size: {os.path.getsize(temp_wav_path)} bytes")

#         if temp_wav_path:
#             is_valid, check_message = check_audio_for_generation(temp_wav_path)
#             if not is_valid:
#                 return check_message, None
#             else:
#                 print(check_message)

#         if not isinstance(wav_pr, torch.Tensor):
#             wav_pr = torch.FloatTensor(wav_pr)

#         if wav_pr.abs().max() > 1:
#             wav_pr = wav_pr / wav_pr.abs().max()

#         if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
#             wav_pr = wav_pr.mean(dim=0, keepdim=True)
#         elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
#             wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)

#         if wav_pr.ndim == 1:
#             wav_pr = wav_pr.unsqueeze(0)

#         assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"

#         text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)

#         if language == 'auto-detect':
#             lang_token = lang2token[langid.classify(text)[0]]
#         else:
#             lang_token = langdropdown2token[language]
#         lang = token2lang[lang_token]
        
#         # Normalize punctuation spacing to reduce pauses
#         text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        
#         # Add language tokens
#         text_with_tokens = lang_token + text + lang_token
        
#         print(f"[Debug] Text with language tokens: {text_with_tokens}")

#         model.to(device)

#         encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
#         audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

#         logging.info(f"synthesize text: {text_with_tokens}")
        
#         # Tokenize with error handling
#         try:
#             # Debug the text being tokenized
#             text_to_tokenize = f"_{text}".strip()
#             print(f"[Debug] Text to tokenize: '{text_to_tokenize}'")
#             phone_tokens, langs = text_tokenizer.tokenize(text=text_to_tokenize)
#             print(f"[Debug] Phone tokens generated: {len(phone_tokens)} tokens")
#             text_tokens, text_tokens_lens = text_collater([phone_tokens])
#         except ValueError as e:
#             if "Empty text is given" in str(e):
#                 model.to('cpu')
#                 torch.cuda.empty_cache()
#                 return f"Error: No valid phonemes generated from text '{text}'. The text may contain only special characters or symbols that cannot be converted to phonemes. Please use regular words and letters.", None
#             else:
#                 model.to('cpu')
#                 torch.cuda.empty_cache()
#                 return f"Error: Failed to process text. Please use simpler text without special characters. Details: {str(e)}", None
#         except Exception as e:
#             model.to('cpu')
#             torch.cuda.empty_cache()
#             return f"Error: Unexpected error during text processing. Details: {str(e)}", None

#         enroll_x_lens = None
#         if text_pr:
#             text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
#             text_prompts, enroll_x_lens = text_collater([text_prompts])

#         text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#         text_tokens_lens += enroll_x_lens
#         lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

#         gen_start = time.time()
#         encoded_frames = model.inference(
#             text_tokens.to(device),
#             text_tokens_lens.to(device),
#             audio_prompts,
#             enroll_x_lens=enroll_x_lens,
#             top_k=-100,
#             temperature=0.7,
#             prompt_language=lang_pr,
#             text_language=langs if accent == "no-accent" else lang,
#             best_of=5,
#         )
#         gen_end = time.time()
        
#         print(f"[Debug] Generation took {gen_end - gen_start:.2f} seconds")

#         frames = encoded_frames.permute(2,0,1)
#         features = vocos.codes_to_features(frames)
#         samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

#         model.to('cpu')
#         torch.cuda.empty_cache()

#         audio_numpy = samples.squeeze(0).cpu().numpy()
        
#         # Normalize audio amplitude
#         max_amp = np.abs(audio_numpy).max()
#         if max_amp > 0:
#             audio_numpy = audio_numpy * (0.95 / max_amp)
        
#         if len(audio_numpy) > 24000 * 300:
#             logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
#             audio_numpy = audio_numpy[:24000 * 300]
        
#         print(f"[Debug] Generated audio duration: {len(audio_numpy)/24000:.2f}s, amplitude range: [{audio_numpy.min():.3f}, {audio_numpy.max():.3f}]")

#         message = f"✓ Generated successfully!\nPrompt: {text_pr}\nSynthesized: {text}\nDuration: {len(audio_numpy)/24000:.1f}s"

#         try:
#             if temp_wav_path and os.path.isfile(temp_wav_path):
#                 print(f"[Debug] Using reference audio for similarity: {temp_wav_path}")
#                 evaluate_and_print_metrics(audio_numpy, 24000, original_text=text_with_tokens, reference_prompt_wav_path=temp_wav_path, save_prefix="infer_from_audio")
#             else:
#                 print(f"[Warning] Reference audio not found: {temp_wav_path}")
#                 evaluate_and_print_metrics(audio_numpy, 24000, original_text=text_with_tokens, reference_prompt_wav_path=None, save_prefix="infer_from_audio")
#         except Exception as e:
#             print(f"[Eval] Evaluation failed: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             # Cleanup temp file after delay
#             if temp_wav_path and temp_wav_path.startswith(os.path.join(os.getcwd(), "temp_enroll_")):
#                 try:
#                     import threading
#                     def delayed_cleanup(path):
#                         time.sleep(120)
#                         try:
#                             if os.path.exists(path):
#                                 os.remove(path)
#                                 print(f"[Cleanup] Removed temp file: {path}")
#                         except:
#                             pass
#                     cleanup_thread = threading.Thread(target=delayed_cleanup, args=(temp_wav_path,))
#                     cleanup_thread.daemon = True
#                     cleanup_thread.start()
#                 except:
#                     pass

#         return message, (24000, audio_numpy)

#     except Exception as e:
#         logging.error(f"Error in infer_from_audio: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         model.to('cpu')
#         torch.cuda.empty_cache()
#         return f"Error: {str(e)}", None

# def main():
#     try:
#         load_speaker_diarization()
#     except Exception as e:
#         print(f"Failed to load speaker diarization: {e}")
#         print("Continuing without speaker diarization...")
    
#     app = gr.Blocks(title="VALL-E X - Voice Cloning")
#     with app:
#         gr.Markdown(top_md)
#         gr.Markdown(infer_from_audio_md)
#         with gr.Row():
#             with gr.Column():
#                 textbox = gr.TextArea(label="Text",
#                                       placeholder="Type your sentence here",
#                                       value="Welcome back, Master. What can I do for you today?", 
#                                       elem_id=f"tts-input")
#                 language_dropdown = gr.Dropdown(choices=['English'], value='English', label='language')
#                 accent_dropdown = gr.Dropdown(choices=['English'], value='English', label='accent')
#                 upload_audio_prompt = gr.Audio(label='Upload audio prompt', interactive=True)
#                 mic_audio_prompt = gr.Audio(source="microphone", label="Record with Microphone", interactive=True)
#             with gr.Column():
#                 text_output = gr.Textbox(label="Message")
#                 audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                 btn = gr.Button("Generate!")
#                 btn.click(
#                     infer_from_audio,
#                     inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, mic_audio_prompt],
#                     outputs=[text_output, audio_output]
#                 )

#     try:
#         webbrowser.open("http://127.0.0.1:7860")
#         app.launch(inbrowser=True)
#     except Exception as e:
#         print(f"Error launching app: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     formatter = (
#         "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
#     )
#     logging.basicConfig(format=formatter, level=logging.INFO)
    
#     try:
#         main()
#     except Exception as e:
#         print(f"Fatal error: {e}")
#         import traceback
#         traceback.print_exc()