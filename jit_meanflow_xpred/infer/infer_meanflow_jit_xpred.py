"""
MeanFlow inference script.
CFM forward_meanflow; supports single/multi-step ODE.
Flow: z_t = (1-t)*x1 + t*x0 (t=0 data, t=1 noise), v = x0 - x1.
Inference: integrate from t=1 to t=0; Euler: x_next = x - (t - t_next) * u.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from scipy.io.wavfile import write
from tqdm import tqdm

# Add project root (repo root) to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omegaconf import OmegaConf
from jit_meanflow_xpred.model import DiT
from jit_meanflow_xpred.model.cfm import CFM
from jit_meanflow_xpred.model.mel_processing import mel_spectrogram_torch_aslp
from jit_meanflow_xpred.infer.utils_infer import load_model

import onnxruntime

# ==================== Default config ====================
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/fm_10ms_contrasive_ecapa_pmeanflow.yaml")
DEFAULT_VOCODER_CONFIG = os.path.join(project_root, "third_party/hifi-gan/config_streamfm10ms.json")
DEFAULT_VOCODER_CKPT = os.path.join(project_root, "third_party/hifi-gan/g_00400000")
DEFAULT_CKPT_FILE = os.path.join(project_root, "ckpt/meanflow/model_last.pt")

# Speaker embedding model (CAMPPlus)
_option = onnxruntime.SessionOptions()
_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
_option.intra_op_num_threads = 1
CAMPPLUS_ONNX = os.path.join(project_root, "third_party/campplus/campplus.onnx")
campplus_session = onnxruntime.InferenceSession(
    CAMPPLUS_ONNX, sess_options=_option, providers=["CPUExecutionProvider"]
)


# ==================== Helpers ====================

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def setup_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _extract_spk_embedding(speech):
    """Extract speaker embedding (CAMPPlus, 16kHz mono input)."""
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(
        None,
        {campplus_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
    )[0].flatten().tolist()
    return torch.tensor([embedding]).cpu().detach()


def load_audio(wav_path, target_sr=16000):
    audio, sr = torchaudio.load(wav_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    mx = audio.abs().max()
    if mx > 0:
        audio = audio / (mx + 1e-6)
    return audio


def load_token(token_path, key=None):
    """Load token from .npy or .jsonl."""
    if token_path.endswith(".npy"):
        return np.load(token_path)
    elif token_path.endswith(".jsonl"):
        assert key is not None, "jsonl format requires --key"
        with open(token_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if str(item.get("key", "")) == str(key):
                    return np.array(item["token"], dtype=np.int64)
        raise KeyError(f"Key '{key}' not found in {token_path}")
    else:
        raise ValueError(f"Unsupported token format: {token_path}")


def hifigan_convert_wav(wav_data, volume_scale=0.5):
    wav_data = wav_data * volume_scale
    audio = wav_data * 32768.0
    return audio.cpu().numpy().astype("int16")


def get_timesteps(steps, device):
    """Time steps from t=1 (noise) to t=0 (data); matches flow z_t = (1-t)*x1 + t*x0."""
    if steps == 1:
        return torch.tensor([1.0, 0.0], device=device)
    elif steps == 2:
        return torch.tensor([1.0, 0.5, 0.0], device=device)
    else:
        return torch.linspace(1.0, 0.0, steps + 1, device=device)


# ==================== Model init ====================

def initialize_model(config_path, ckpt_file, device):
    """Initialize CFM + DiT model."""
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.arch
    vocab_size = cfg.model.text_num_embeds

    ema_model = load_model(
        DiT,
        model_cfg,
        ckpt_file,
        vocab_size=vocab_size,
        mel_spec_type="hifigan",
        vocab_file="",
        device=device,
    )
    return ema_model


def initialize_vocoder(vocoder_config_path, vocoder_ckpt_path, device):
    """Initialize HiFi-GAN vocoder."""
    hifigan_src = os.path.join(project_root, "third_party/hifi-gan/src")
    if hifigan_src not in sys.path:
        sys.path.append(hifigan_src)
    from third_party.hifigan.models import Generator

    with open(vocoder_config_path) as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device)
    ckpt = torch.load(vocoder_ckpt_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator


# ==================== Inference ====================

@torch.inference_mode()
def inference_meanflow(
    model: CFM,
    vocoder,
    token,          # [B, token_len]  int64
    spk_emb,        # [B, 192]
    prompt_mel,     # [B, T_ref, 80] reference mel
    steps=1,
    cfg_strength=2.0,
    device="cuda",
    output_wav_path=None,
    chunk_size=0,   # 0 = full sequence, >0 = mel frames per chunk
):
    """
    MeanFlow inference. ODE from t=1 to t=0; Euler step: x_next = x_t - dt * u(x_t, t).
    Returns wav tensor and inference time in seconds.
    """
    model.eval()

    B = token.shape[0]
    token_len = token.shape[1]

    # Token-to-mel: 4 frames per token (10ms mel, 40ms token)
    mel_len = token_len * 4
    mel_dim = 80

    timesteps = get_timesteps(steps, device)

    # ---------- Full-sequence (no chunking) ----------
    if chunk_size <= 0:
        x = torch.randn(B, mel_len, mel_dim, device=device, dtype=prompt_mel.dtype)
        start_time = time.time()

        for i in range(steps):
            t_cur = timesteps[i]
            t_nxt = timesteps[i + 1]
            dt = t_cur - t_nxt
            t_tensor = torch.full((B,), t_cur.item(), device=device)
            x_pred = model.transformer(
                x=x,
                cond=prompt_mel,
                text=token,
                spk_emb=spk_emb,
                time=t_tensor,
                drop_audio_cond=False,
                drop_text=False,
            )
            u = (x - x_pred) / (t_tensor + 1e-8)
            x = x - dt * u

        inference_time = time.time() - start_time

    # ---------- Chunked inference ----------
    else:
        x_chunks = []
        start_time = time.time()

        for seg_start in range(0, mel_len, chunk_size):
            seg_end = min(seg_start + chunk_size, mel_len)
            chunk_len = seg_end - seg_start

            tok_start = seg_start // 4
            tok_end = min(math.ceil(seg_end / 4), token_len)
            token_chunk = token[:, tok_start:tok_end]
            x = torch.randn(B, chunk_len, mel_dim, device=device, dtype=prompt_mel.dtype)

            for i in range(steps):
                t_cur = timesteps[i]
                t_nxt = timesteps[i + 1]
                dt = t_cur - t_nxt

                t_tensor = torch.full((B,), t_cur.item(), device=device)

                x_pred = model.transformer(
                    x=x,
                    cond=prompt_mel,
                    text=token_chunk,
                    spk_emb=spk_emb,
                    time=t_tensor,
                    drop_audio_cond=False,
                    drop_text=False,
                )

                u = (x - x_pred) / (t_tensor + 1e-8)
                x = x - dt * u

            x_chunks.append(x)

        x = torch.cat(x_chunks, dim=1)  # [B, mel_len, 80]
        inference_time = time.time() - start_time

    # ---------- Vocoder ----------
    mel = x.permute(0, 2, 1).float()  # [B, 80, T]
    wav = vocoder(mel)  # [B, 1, T_wav]

    # ---------- Save ----------
    if output_wav_path is not None:
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        wav_int16 = hifigan_convert_wav(wav.squeeze())
        write(output_wav_path, 24000, wav_int16)
        print(f"Saved: {output_wav_path}")

    return wav, inference_time


# Token-to-mel ratio: 4 mel frames per token (10ms mel, 40ms token)
TOKEN_MEL_RATIO = 4


@torch.inference_mode()
def inference_meanflow_one_chunk(
    model: CFM,
    vocoder,
    token_chunk,
    spk_emb,
    prompt_mel,
    output_tokens=18,
    steps=1,
    cfg_strength=2.0,
    device="cuda",
):
    """
    Run MeanFlow for one token chunk (e.g. 24 tokens) and return wav for the first
    output_tokens only (e.g. 18). Used for streaming: 24-token context with 6-token
    lookahead, output 18 tokens per step.
    """
    model.eval()
    B = token_chunk.shape[0]
    chunk_len = token_chunk.shape[1]
    mel_chunk_frames = chunk_len * TOKEN_MEL_RATIO
    mel_dim = 80
    timesteps = get_timesteps(steps, device)

    x = torch.randn(B, mel_chunk_frames, mel_dim, device=device, dtype=prompt_mel.dtype)
    for i in range(steps):
        t_cur = timesteps[i]
        t_nxt = timesteps[i + 1]
        dt = t_cur - t_nxt
        t_tensor = torch.full((B,), t_cur.item(), device=device)
        x_pred = model.transformer(
            x=x,
            cond=prompt_mel,
            text=token_chunk,
            spk_emb=spk_emb,
            time=t_tensor,
            drop_audio_cond=False,
            drop_text=False,
        )
        u = (x - x_pred) / (t_tensor + 1e-8)
        x = x - dt * u

    mel_full = x.permute(0, 2, 1).float()
    out_frames = min(output_tokens * TOKEN_MEL_RATIO, mel_full.shape[2])
    mel_out = mel_full[:, :, :out_frames]
    wav = vocoder(mel_out)
    return wav


def inference_meanflow_streaming(
    model: CFM,
    vocoder,
    token,
    spk_emb,
    prompt_mel,
    first_chunk_tokens=24,
    hop_tokens=18,
    lookahead_tokens=6,
    steps=1,
    cfg_strength=2.0,
    device="cuda",
):
    """
    Streaming inference: first chunk 24 tokens, then each chunk 24 tokens (18 new + 6 lookahead),
    advance by 18 tokens per step. Yields wav chunks for 18 tokens each (except possibly the last).
    """
    B, token_len = token.shape[0], token.shape[1]
    context_len = first_chunk_tokens
    offset = 0
    while offset + context_len <= token_len:
        chunk = token[:, offset : offset + context_len]
        wav = inference_meanflow_one_chunk(
            model=model,
            vocoder=vocoder,
            token_chunk=chunk,
            spk_emb=spk_emb,
            prompt_mel=prompt_mel,
            output_tokens=hop_tokens,
            steps=steps,
            cfg_strength=cfg_strength,
            device=device,
        )
        yield wav
        offset += hop_tokens
    if offset < token_len:
        remainder = token[:, offset:]
        wav, _ = inference_meanflow(
            model=model,
            vocoder=vocoder,
            token=remainder,
            spk_emb=spk_emb,
            prompt_mel=prompt_mel,
            steps=steps,
            cfg_strength=cfg_strength,
            device=device,
            chunk_size=0,
        )
        yield wav


# ==================== Single-sample inference ====================

def run_single_inference(
    model,
    vocoder,
    prompt_wav_path,
    token_path,
    output_dir,
    steps=1,
    cfg_strength=2.0,
    chunk_size=0,
    device="cuda",
    key=None,
):
    """Full pipeline: load token -> load ref audio -> extract spk_emb + mel -> infer -> save."""
    # Load token
    token = load_token(token_path, key=key)
    token = torch.from_numpy(token).unsqueeze(0).to(device)  # [1, L]

    # Load reference audio (16kHz)
    audio = load_audio(prompt_wav_path, target_sr=16000)

    # Speaker embedding
    spk_emb = _extract_spk_embedding(audio).to(device)  # [1, 192]

    # Reference mel (cond)
    mel_spec = mel_spectrogram_torch_aslp(
        y=audio, n_fft=1024, num_mels=80, sampling_rate=16000,
        hop_size=160, win_size=640, fmin=0, fmax=8000, center=False,
    )
    prompt_mel = mel_spec.permute(0, 2, 1).to(device)  # [1, T, 80]

    # Output path
    basename = os.path.splitext(os.path.basename(token_path))[0]
    if key is not None:
        basename = str(key)
    mode_tag = f"meanflow_steps{steps}_cfg{cfg_strength}"
    output_wav_path = os.path.join(output_dir, mode_tag, f"{basename}.wav")

    wav, infer_time = inference_meanflow(
        model=model,
        vocoder=vocoder,
        token=token,
        spk_emb=spk_emb,
        prompt_mel=prompt_mel,
        steps=steps,
        cfg_strength=cfg_strength,
        device=device,
        output_wav_path=output_wav_path,
        chunk_size=chunk_size,
    )

    # RTF
    duration = wav.shape[-1] / 24000
    rtf = infer_time / duration if duration > 0 else 0
    print(f"[{basename}] steps={steps}  time={infer_time:.3f}s  "
          f"duration={duration:.2f}s  RTF={rtf:.4f}")

    return wav, infer_time


def run_single_inference_streaming(
    model,
    vocoder,
    prompt_wav_path,
    token_path,
    output_dir,
    first_chunk_tokens=24,
    hop_tokens=18,
    steps=1,
    cfg_strength=2.0,
    device="cuda",
    key=None,
):
    """Single-sample inference using streaming (24-token first chunk, 18-token hop, 6-token lookahead)."""
    token = load_token(token_path, key=key)
    token = torch.from_numpy(token).unsqueeze(0).to(device)

    audio = load_audio(prompt_wav_path, target_sr=16000)
    spk_emb = _extract_spk_embedding(audio).to(device)
    mel_spec = mel_spectrogram_torch_aslp(
        y=audio, n_fft=1024, num_mels=80, sampling_rate=16000,
        hop_size=160, win_size=640, fmin=0, fmax=8000, center=False,
    )
    prompt_mel = mel_spec.permute(0, 2, 1).to(device)

    basename = os.path.splitext(os.path.basename(token_path))[0]
    if key is not None:
        basename = str(key)
    mode_tag = f"meanflow_stream_steps{steps}_cfg{cfg_strength}"
    output_wav_path = os.path.join(output_dir, mode_tag, f"{basename}.wav")

    start_time = time.time()
    chunks = list(inference_meanflow_streaming(
        model=model,
        vocoder=vocoder,
        token=token,
        spk_emb=spk_emb,
        prompt_mel=prompt_mel,
        first_chunk_tokens=first_chunk_tokens,
        hop_tokens=hop_tokens,
        lookahead_tokens=6,
        steps=steps,
        cfg_strength=cfg_strength,
        device=device,
    ))
    infer_time = time.time() - start_time
    wav = torch.cat(chunks, dim=-1)
    duration = wav.shape[-1] / 24000
    rtf = infer_time / duration if duration > 0 else 0

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    wav_int16 = hifigan_convert_wav(wav.squeeze())
    write(output_wav_path, 24000, wav_int16)
    print(f"Saved: {output_wav_path}")
    print(f"[{basename}] streaming  chunks={len(chunks)}  time={infer_time:.3f}s  duration={duration:.2f}s  RTF={rtf:.4f}")
    return wav, infer_time


# ==================== JSONL batch inference ====================

def run_jsonl_all_inference(
    model,
    vocoder,
    prompt_wav_path,
    token_path,
    output_dir,
    steps=1,
    cfg_strength=2.0,
    chunk_size=0,
    device="cuda",
):
    """
    When token_path is .jsonl and --key is not set, run inference for each line.
    If prompt_wav_path is a dir: use <prompt_wav_path>/<key>.wav per line.
    If prompt_wav_path is a single wav: reuse the same reference for all.
    JSONL format: {"key": "xxx", "token": [...]} or {"key": "xxx", "code": [...]}.
    """
    assert token_path.endswith(".jsonl"), "run_jsonl_all_inference requires .jsonl"

    use_dir_prompt = os.path.isdir(prompt_wav_path)

    # Single ref: precompute cond; dir mode: load per key in loop
    if not use_dir_prompt:
        audio = load_audio(prompt_wav_path, target_sr=16000)
        spk_emb_global = _extract_spk_embedding(audio).to(device)  # [1, 192]
        mel_spec_global = mel_spectrogram_torch_aslp(
            y=audio, n_fft=1024, num_mels=80, sampling_rate=16000,
            hop_size=160, win_size=640, fmin=0, fmax=8000, center=False,
        )
        prompt_mel_global = mel_spec_global.permute(0, 2, 1).to(device)  # [1, T, 80]

    mode_tag = f"meanflow_steps{steps}_cfg{cfg_strength}"
    out_dir_mode = os.path.join(output_dir, mode_tag)
    os.makedirs(out_dir_mode, exist_ok=True)

    all_times = []
    all_durations = []

    with open(token_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = str(item.get("key", ""))
            if not key:
                continue
            # Prefer "token" field, then "code"
            if "token" in item:
                token_seq = item["token"]
            elif "code" in item:
                token_seq = item["code"]
            else:
                print(f"  skip: no 'token' or 'code' field for key {key}")
                continue
            token_np = np.array(token_seq, dtype=np.int64)
            token = torch.from_numpy(token_np).unsqueeze(0).to(device)

            output_wav_path = os.path.join(out_dir_mode, f"{key}.wav")

            # Choose ref audio by mode
            if use_dir_prompt:
                wav_path = os.path.join(prompt_wav_path, f"{key}.wav")
                if not os.path.exists(wav_path):
                    print(f"  skip: prompt wav not found for key {key}: {wav_path}")
                    continue
                audio = load_audio(wav_path, target_sr=16000)
                spk_emb = _extract_spk_embedding(audio).to(device)  # [1, 192]
                mel_spec = mel_spectrogram_torch_aslp(
                    y=audio, n_fft=1024, num_mels=80, sampling_rate=16000,
                    hop_size=160, win_size=640, fmin=0, fmax=8000, center=False,
                )
                prompt_mel = mel_spec.permute(0, 2, 1).to(device)  # [1, T, 80]
            else:
                spk_emb = spk_emb_global
                prompt_mel = prompt_mel_global

            wav, infer_time = inference_meanflow(
                model=model,
                vocoder=vocoder,
                token=token,
                spk_emb=spk_emb,
                prompt_mel=prompt_mel,
                steps=steps,
                cfg_strength=cfg_strength,
                device=device,
                output_wav_path=output_wav_path,
                chunk_size=chunk_size,
            )

            duration = wav.shape[-1] / 24000
            rtf = infer_time / duration if duration > 0 else 0
            print(f"[{key}] steps={steps}  time={infer_time:.3f}s  "
                  f"duration={duration:.2f}s  RTF={rtf:.4f}")

            all_times.append(infer_time)
            all_durations.append(duration)

    if all_durations:
        total_t = sum(all_times)
        total_d = sum(all_durations)
        rtf = total_t / total_d if total_d > 0 else 0
        print(f"\n{'='*55}")
        print(f" MeanFlow jsonl All Inference Summary  (steps={steps})")
        print(f"{'='*55}")
        print(f" Samples       : {len(all_times)}")
        print(f" Infer time    : {total_t:.2f}s")
        print(f" Audio duration: {total_d:.2f}s")
        print(f" RTF           : {rtf:.4f}")
        print(f"{'='*55}")


# ==================== Batch inference ====================

def run_batch_inference(
    model,
    vocoder,
    prompt_wav_dir,
    token_path,
    output_dir,
    steps=1,
    cfg_strength=2.0,
    chunk_size=0,
    device="cuda",
):
    """
    Batch inference: iterate all .wav in prompt_wav_dir and match tokens.
    token_path: directory (with .npy files) or single .jsonl file.
    """
    wav_files = sorted(glob.glob(os.path.join(prompt_wav_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {prompt_wav_dir}")
        return

    # Token source
    is_jsonl = token_path.endswith(".jsonl")
    token_dict = {}
    if is_jsonl:
        with open(token_path) as f:
            for line in f:
                item = json.loads(line.strip())
                token_dict[str(item["key"])] = np.array(item["token"], dtype=np.int64)
        print(f"Loaded {len(token_dict)} tokens from jsonl")

    all_times = []
    all_durations = []

    for wav_path in tqdm(wav_files, desc="Inference"):
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        # Match token
        if is_jsonl:
            if basename not in token_dict:
                print(f"  skip: key '{basename}' not in jsonl")
                continue
            token = token_dict[basename]
            token = torch.from_numpy(token).unsqueeze(0).to(device)
        else:
            npy_path = os.path.join(token_path, f"{basename}.npy")
            if not os.path.exists(npy_path):
                npy_path = os.path.join(token_path, f"{basename}.hubert_code.npy")
            if not os.path.exists(npy_path):
                print(f"  skip: token not found for {basename}")
                continue
            token = torch.from_numpy(np.load(npy_path)).unsqueeze(0).to(device)

        try:
            audio = load_audio(wav_path, target_sr=16000)
            spk_emb = _extract_spk_embedding(audio).to(device)
            mel_spec = mel_spectrogram_torch_aslp(
                y=audio, n_fft=1024, num_mels=80, sampling_rate=16000,
                hop_size=160, win_size=640, fmin=0, fmax=8000, center=False,
            )
            prompt_mel = mel_spec.permute(0, 2, 1).to(device)

            mode_tag = f"meanflow_steps{steps}_cfg{cfg_strength}"
            out_path = os.path.join(output_dir, mode_tag, f"{basename}.wav")

            wav, infer_time = inference_meanflow(
                model=model, vocoder=vocoder,
                token=token, spk_emb=spk_emb, prompt_mel=prompt_mel,
                steps=steps, cfg_strength=cfg_strength,
                device=device, output_wav_path=out_path,
                chunk_size=chunk_size,
            )

            dur = wav.shape[-1] / 24000
            all_times.append(infer_time)
            all_durations.append(dur)

        except Exception as e:
            print(f"  Error {basename}: {e}")
            continue

    # Summary
    if all_durations:
        total_t = sum(all_times)
        total_d = sum(all_durations)
        rtf = total_t / total_d if total_d > 0 else 0
        print(f"\n{'='*55}")
        print(f" MeanFlow Inference Summary  (steps={steps})")
        print(f"{'='*55}")
        print(f" Samples       : {len(all_times)}")
        print(f" Infer time    : {total_t:.2f}s")
        print(f" Audio duration: {total_d:.2f}s")
        print(f" RTF           : {rtf:.4f}")
        print(f"{'='*55}")


# ==================== CLI ====================

def parse_args():
    p = argparse.ArgumentParser(description="MeanFlow TTS Inference")
    p.add_argument("--prompt_wav", type=str, required=True, help="Reference audio path or directory")
    p.add_argument("--token_path", type=str, required=True, help="Token: .npy / .jsonl / dir")
    p.add_argument("--key", type=str, default=None, help="Key for jsonl (single-sample)")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--ckpt_file", type=str, default=DEFAULT_CKPT_FILE)
    p.add_argument("--vocoder_config", type=str, default=DEFAULT_VOCODER_CONFIG)
    p.add_argument("--vocoder_ckpt", type=str, default=DEFAULT_VOCODER_CKPT)
    p.add_argument("--steps", type=int, default=1, help="ODE steps (1 = MeanFlow single step)")
    p.add_argument("--cfg_strength", type=float, default=2.0, help="CFG strength; 0 = no CFG")
    p.add_argument("--chunk_size", type=int, default=0, help="Mel chunk size; 0 = full sequence")
    p.add_argument("--batch", action="store_true", help="Batch inference")
    p.add_argument("--stream", action="store_true", help="Streaming: first 24 tokens, then 18-token hop with 6-token lookahead")

    # Other
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MeanFlow model ...")
    model = initialize_model(args.config_path, args.ckpt_file, device)

    print("Loading HiFi-GAN vocoder ...")
    vocoder = initialize_vocoder(args.vocoder_config, args.vocoder_ckpt, device)

    if args.batch:
        run_batch_inference(
            model=model, vocoder=vocoder,
            prompt_wav_dir=args.prompt_wav,
            token_path=args.token_path,
            output_dir=args.output_dir,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            chunk_size=args.chunk_size,
            device=device,
        )
    elif args.token_path.endswith(".jsonl") and args.key is None:
        run_jsonl_all_inference(
            model=model,
            vocoder=vocoder,
            prompt_wav_path=args.prompt_wav,
            token_path=args.token_path,
            output_dir=args.output_dir,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            chunk_size=args.chunk_size,
            device=device,
        )
    else:
        if getattr(args, "stream", False):
            run_single_inference_streaming(
                model=model,
                vocoder=vocoder,
                prompt_wav_path=args.prompt_wav,
                token_path=args.token_path,
                output_dir=args.output_dir,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                device=device,
                key=args.key,
            )
        else:
            run_single_inference(
                model=model, vocoder=vocoder,
                prompt_wav_path=args.prompt_wav,
                token_path=args.token_path,
                output_dir=args.output_dir,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                chunk_size=args.chunk_size,
                device=device,
                key=args.key,
            )


if __name__ == "__main__":
    main()