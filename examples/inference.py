# Copyright (c) 2024 Alibaba Inc (authors: Contributors)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashTTS inference example.
Supports: (1) FlashTTS text-to-speech with voice cloning;
          (2) MeanFlow-only mode (token file + reference wav -> wav), including streaming.
Run from repo root: python examples/inference.py --mode flash_tts --text "Hello" --prompt_wav ref.wav --output_dir out
"""

from __future__ import print_function

import argparse
import logging
import os
import sys

# Add repo root so cosyvoice and jit_meanflow_xpred can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_args():
    p = argparse.ArgumentParser(description="FlashTTS inference")
    p.add_argument("--mode", default="flash_tts", choices=["flash_tts", "meanflow_only"],
                   help="flash_tts: text + prompt_wav -> wav (uses FlashTTS). "
                        "meanflow_only: token file + prompt_wav -> wav (uses jit_meanflow_xpred only).")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Model directory (for flash_tts: dir with llm.pt and config; for meanflow_only not used)")
    p.add_argument("--pretrained_model_dir", type=str, default=None,
                   help="Pretrained/base model dir (flash_tts: yaml, MeanFlow/vocoder assets; meanflow_only: not used)")
    p.add_argument("--text", type=str, default="",
                   help="Text to synthesize (flash_tts mode)")
    p.add_argument("--prompt_wav", type=str, required=True, help="Reference audio path (16 kHz mono)")
    p.add_argument("--token_path", type=str, default=None,
                   help="Token file path for meanflow_only mode (.npy or .jsonl with --key)")
    p.add_argument("--key", type=str, default=None, help="Key for jsonl in meanflow_only mode")
    p.add_argument("--output_dir", type=str, default="./inference_output", help="Output directory")
    p.add_argument("--output_name", type=str, default="out", help="Output wav base name (no extension)")
    p.add_argument("--stream", action="store_true", help="Enable streaming (flash_tts or meanflow_only)")
    p.add_argument("--steps", type=int, default=1, help="MeanFlow ODE steps (meanflow_only)")
    p.add_argument("--config_path", type=str, default=None, help="MeanFlow config (meanflow_only)")
    p.add_argument("--ckpt_file", type=str, default=None, help="MeanFlow checkpoint (meanflow_only)")
    p.add_argument("--vocoder_config", type=str, default=None, help="Vocoder config (meanflow_only)")
    p.add_argument("--vocoder_ckpt", type=str, default=None, help="Vocoder checkpoint (meanflow_only)")
    p.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    return p.parse_args()


def run_flash_tts(args):
    """Text + prompt_wav -> wav using FlashTTS."""
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import FlashTTS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    model_dir = args.model_dir or os.path.join(_REPO_ROOT, "pretrained_models", "FlashTTS-0.5B")
    pretrained_dir = args.pretrained_model_dir or model_dir
    if not os.path.exists(model_dir):
        logging.warning("model_dir %s not found; you may need to set --model_dir and --pretrained_model_dir", model_dir)

    model = FlashTTS(model_dir, pretrained_model_dir=pretrained_dir, fp16=torch.cuda.is_available())
    sample_rate = model.sample_rate
    text = args.text or "Hello, this is a FlashTTS example."
    # Load prompt audio: frontend expects 16 kHz mono tensor [1, samples]
    prompt_wav, sr = torchaudio.load(args.prompt_wav)
    if prompt_wav.shape[0] > 1:
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        prompt_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(prompt_wav)
    prompt_speech_16k = prompt_wav.unsqueeze(0) if prompt_wav.dim() == 1 else prompt_wav  # [1, T]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.wav")

    chunks = []
    for out in model.inference_zero_shot(text, "", prompt_speech_16k, stream=args.stream):
        chunks.append(out["tts_speech"])

    wav = torch.cat(chunks, dim=1)
    torchaudio.save(out_path, wav.cpu(), sample_rate=sample_rate)
    logging.info("Saved: %s", out_path)
    return out_path


def run_meanflow_only(args):
    """Token file + prompt_wav -> wav using jit_meanflow_xpred (optionally with streaming)."""
    import torch
    from jit_meanflow_xpred.infer.infer_meanflow_jit_xpred import (
        initialize_model,
        initialize_vocoder,
        run_single_inference,
        run_single_inference_streaming,
        DEFAULT_CONFIG_PATH,
        DEFAULT_CKPT_FILE,
        DEFAULT_VOCODER_CONFIG,
        DEFAULT_VOCODER_CKPT,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    config_path = args.config_path or DEFAULT_CONFIG_PATH
    ckpt_file = args.ckpt_file or DEFAULT_CKPT_FILE
    vocoder_config = args.vocoder_config or DEFAULT_VOCODER_CONFIG
    vocoder_ckpt = args.vocoder_ckpt or DEFAULT_VOCODER_CKPT

    model = initialize_model(config_path, ckpt_file, device)
    vocoder = initialize_vocoder(vocoder_config, vocoder_ckpt, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.stream:
        wav, _ = run_single_inference_streaming(
            model=model,
            vocoder=vocoder,
            prompt_wav_path=args.prompt_wav,
            token_path=args.token_path,
            output_dir=args.output_dir,
            steps=args.steps,
            device=device,
            key=args.key,
        )
    else:
        wav, _ = run_single_inference(
            model=model,
            vocoder=vocoder,
            prompt_wav_path=args.prompt_wav,
            token_path=args.token_path,
            output_dir=args.output_dir,
            steps=args.steps,
            device=device,
            key=args.key,
        )

    logging.info("MeanFlow inference done; output written under %s", args.output_dir)
    return args.output_dir


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "flash_tts":
        run_flash_tts(args)
    elif args.mode == "meanflow_only":
        if not args.token_path:
            raise ValueError("meanflow_only mode requires --token_path")
        run_meanflow_only(args)
    else:
        raise ValueError("Unknown mode: %s", args.mode)


if __name__ == "__main__":
    main()
