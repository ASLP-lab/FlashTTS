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
import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper
from loguru import logger
# Import jit_meanflow_xpred for token2mel + vocoder (MeanFlow backend)
try:
    import sys
    import os as _os_path
    _project_root = _os_path.abspath(_os_path.join(_os_path.dirname(__file__), "../../"))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from jit_meanflow_xpred.infer.infer_meanflow_jit_xpred import (
        inference_meanflow,
        inference_meanflow_one_chunk,
    )
    HAS_MEANFLOW = True
except ImportError:
    HAS_MEANFLOW = False
    inference_meanflow = None
    inference_meanflow_one_chunk = None
    logger.warning("MeanFlow module not available")
    

class FlashTTS:
    """FlashTTS: LLM + MeanFlow (token2mel) + vocoder. backbone."""

    def __init__(self, llm: torch.nn.Module, meanflow_model, vocoder, fp16: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.meanflow_model = meanflow_model
        self.vocoder = vocoder
        self.fp16 = fp16
        if self.fp16:
            self.llm.half()
        self.first_chunk_tokens = 24
        self.token_hop_len = 18
        self.pre_lookahead_len = 6
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.llm_first_token_time_s = {}
        self.llm_token_speed_metrics = {}

    def load(self, llm_checkpoint):
        self.llm.load_state_dict(torch.load(llm_checkpoint, map_location=self.device), strict=False)
        self.llm.to(self.device).eval()

    def load_vllm(self, model_dir):
        from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        engine_args = EngineArgs(model=model_dir, skip_tokenizer_init=True, enable_prompt_embeds=True, gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, spk_embedding, uuid, is_from_speech, is_from_text, cfg_drop=0.0, is_tokenlevel=False, is_icl=False):
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 and not getattr(self.llm, "vllm", None)):
            if isinstance(text, Generator):
                for i in self.llm.inference_bistream(
                    text=text, prompt_text=prompt_text.to(self.device),
                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                    embedding=llm_embedding.to(self.device), spk_embedding=spk_embedding.to(self.device),
                    is_tokenlevel=is_tokenlevel,
                ):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                if is_from_speech or is_from_text:
                    for i in self.llm.inference_with_spkemb(
                        text=text.to(self.device), text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_text=prompt_text.to(self.device), prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device), prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device), spk_embedding=spk_embedding.to(self.device),
                        uuid=uuid, is_from_speech=is_from_speech, is_from_text=is_from_text, cfg_drop=cfg_drop, is_tokenlevel=is_tokenlevel, is_icl=is_icl,
                    ):
                        self.tts_speech_token_dict[uuid].append(i)
                        if hasattr(self, "llm_first_token_time_s") and len(self.tts_speech_token_dict[uuid]) == 1:
                            with self.lock:
                                self.llm_first_token_time_s[uuid] = getattr(self.llm, "_first_token_time_s", None)
                        if hasattr(self, "llm_first_token_time_s") and len(self.tts_speech_token_dict[uuid]) == 18:
                            with self.lock:
                                self.llm_first_token_time_s[uuid] = getattr(self.llm, "_first_18_token_time_s", None)
                        if hasattr(self, "llm_token_speed_metrics") and len(self.tts_speech_token_dict[uuid]) >= 18:
                            with self.lock:
                                self.llm_token_speed_metrics[uuid] = {
                                    "first_18_token_time_s": getattr(self.llm, "_first_18_token_time_s", None),
                                    "every_12_token_time_s_list": getattr(self.llm, "_every_12_token_time_s_list", []),
                                }
                else:
                    for i in self.llm.inference_ori(
                        text=text.to(self.device), text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_text=prompt_text.to(self.device), prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device), prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device), uuid=uuid,
                    ):
                        self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0, profile_timing=False):
        if not HAS_MEANFLOW or inference_meanflow is None:
            raise RuntimeError("MeanFlow backend not available")
        prompt_mel = prompt_feat.to(self.device)
        spk_emb = embedding.to(self.device) if embedding.numel() else torch.zeros(1, 192, device=self.device, dtype=prompt_mel.dtype)
        token_i = token.to(self.device)
        if stream and not finalize and token_i.shape[1] >= self.first_chunk_tokens:
            if inference_meanflow_one_chunk is not None:
                wav = inference_meanflow_one_chunk(
                    model=self.meanflow_model,
                    vocoder=self.vocoder,
                    token_chunk=token_i[:, : self.first_chunk_tokens],
                    spk_emb=spk_emb,
                    prompt_mel=prompt_mel,
                    output_tokens=self.token_hop_len,
                    steps=1,
                    cfg_strength=2.0,
                    device=self.device,
                )
            else:
                wav, _ = inference_meanflow(
                    model=self.meanflow_model,
                    vocoder=self.vocoder,
                    token=token_i,
                    spk_emb=spk_emb,
                    prompt_mel=prompt_mel,
                    steps=1,
                    cfg_strength=2.0,
                    device=self.device,
                    chunk_size=0,
                )
        else:
            wav, _ = inference_meanflow(
                model=self.meanflow_model,
                vocoder=self.vocoder,
                token=token_i,
                spk_emb=spk_emb,
                prompt_mel=prompt_mel,
                steps=1,
                cfg_strength=2.0,
                device=self.device,
                chunk_size=0,
            )
        if speed != 1.0 and finalize:
            wav = F.interpolate(wav, size=int(wav.shape[-1] / speed), mode="linear")
        return wav

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), spk_embedding=torch.zeros(0, 192),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0,
            cfg_drop=0.0, is_tokenlevel=False, is_icl=False, profile_timing=False, **kwargs):
        is_from_speech = kwargs.get("is_from_speech", False)
        is_from_text = kwargs.get("is_from_text", False)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.llm_first_token_time_s[this_uuid] = None
            self.llm_token_speed_metrics[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(
                target=self.llm_job,
                args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, spk_embedding, this_uuid, is_from_speech, is_from_text, cfg_drop, is_tokenlevel, is_icl),
            )
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream:
            token_offset = 0
            context_len = self.first_chunk_tokens
            while True:
                time.sleep(0.01)
                num_tokens = len(self.tts_speech_token_dict[this_uuid])
                need = token_offset + context_len
                if num_tokens >= need:
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][token_offset:need]
                    ).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        uuid=this_uuid,
                        stream=True,
                        finalize=False,
                        profile_timing=profile_timing,
                    )
                    token_offset += self.token_hop_len
                    yield {"tts_speech": this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] and num_tokens < need:
                    break
            p.join()
            remainder = self.tts_speech_token_dict[this_uuid]
            if remainder:
                this_tts_speech_token = torch.tensor(remainder).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(
                    token=this_tts_speech_token,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    embedding=flow_embedding,
                    token_offset=token_offset,
                    uuid=this_uuid,
                    finalize=True,
                    profile_timing=profile_timing,
                )
                yield {"tts_speech": this_tts_speech.cpu()}
        else:
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=0,
                uuid=this_uuid,
                finalize=True,
                speed=speed,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid, None)
            self.llm_end_dict.pop(this_uuid, None)
            self.hift_cache_dict.pop(this_uuid, None)
            self.llm_first_token_time_s.pop(this_uuid, None)
            self.llm_token_speed_metrics.pop(this_uuid, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        unexpected_keys = self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=False)
        logger.debug(f"unexpected_keys: {unexpected_keys}")
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, spk_embedding, uuid, is_from_speech, is_from_text, cfg_drop=0.0, is_tokenlevel=False, is_icl=False):
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model) and not hasattr(self.llm, 'vllm'), 'streaming input text is only implemented for CosyVoice2 and do not support vllm!'
                print("text", type(text))
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device),
                                                     spk_embedding=spk_embedding.to(self.device),
                                                     is_tokenlevel=is_tokenlevel):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                if is_from_speech is True or is_from_text is True:
                    if False and is_icl is True:
                        for i in self.llm.inference_with_icl_spkemb(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                spk_embedding=spk_embedding.to(self.device),
                                                is_tokenlevel=is_tokenlevel,
                                                is_icl=is_icl):
                            self.tts_speech_token_dict[uuid].append(i)
                    else:
                        # text = text[:,:2]
                        
                        for i in self.llm.inference_with_spkemb(text=text.to(self.device),
                                                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                    prompt_text=prompt_text.to(self.device),
                                                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                    prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                    prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                    embedding=llm_embedding.to(self.device),
                                                    spk_embedding=spk_embedding.to(self.device),
                                                    uuid=uuid, is_from_speech=is_from_speech, is_from_text=is_from_text, cfg_drop=cfg_drop, is_tokenlevel=is_tokenlevel, is_icl=is_icl):
                            self.tts_speech_token_dict[uuid].append(i)
                            if hasattr(self, 'llm_first_token_time_s') and len(self.tts_speech_token_dict[uuid]) == 1:
                                with self.lock:
                                    self.llm_first_token_time_s[uuid] = getattr(self.llm, '_first_token_time_s', None)
                            if hasattr(self, 'llm_first_token_time_s') and len(self.tts_speech_token_dict[uuid]) == 18:
                                with self.lock:
                                    self.llm_first_token_time_s[uuid] = getattr(self.llm, '_first_18_token_time_s', None)
                            if hasattr(self, 'llm_token_speed_metrics') and len(self.tts_speech_token_dict[uuid]) >= 18:
                                with self.lock:
                                    self.llm_token_speed_metrics[uuid] = {
                                        'first_18_token_time_s': getattr(self.llm, '_first_18_token_time_s', None),
                                        'every_12_token_time_s_list': getattr(self.llm, '_every_12_token_time_s_list', []),
                                    }
                else:
                    for i in self.llm.inference_ori(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                uuid=uuid):
                        self.tts_speech_token_dict[uuid].append(i)

        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            spk_embedding=torch.zeros(0, 192),
            spk_embedding_drop=None,
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, 
            cfg_drop=0.0, is_tokenlevel=False, is_icl=False, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        is_from_speech = kwargs.get('is_from_speech', False)
        is_from_text = kwargs.get('is_from_text', False)
        is_tokenlevel = kwargs.get('is_tokenlevel', False)
        is_icl = kwargs.get('is_icl', False)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, spk_embedding, this_uuid, is_from_speech, is_from_text, cfg_drop, is_tokenlevel, is_icl))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.llm_first_token_time_s = {}
        self.llm_token_speed_metrics = {}

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0, profile_timing=False):
        if profile_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_flow_start = time.perf_counter()
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        flow_s = (time.perf_counter() - t_flow_start) if profile_timing else 0
        if profile_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        t_hift_start = time.perf_counter()
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        hift_s = (time.perf_counter() - t_hift_start) if profile_timing else 0
        if profile_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if not hasattr(self, '_token2wav_timings'):
                self._token2wav_timings = {}
            self._token2wav_timings[uuid] = {'flow_s': flow_s, 'hift_s': hift_s, 'token2wav_s': flow_s + hift_s}
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            spk_embedding=torch.zeros(0, 192),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, cfg_drop=0.0, is_tokenlevel=False, is_icl=False, profile_timing=False, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        is_from_speech = kwargs.get('is_from_speech', False)
        is_from_text = kwargs.get('is_from_text', False)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.llm_first_token_time_s[this_uuid] = None
            self.llm_token_speed_metrics[this_uuid] = None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, spk_embedding, this_uuid, is_from_speech, is_from_text, cfg_drop, is_tokenlevel, is_icl))
        p.start()   
        self.token_hop_len = 18
        if stream is True:
            token_offset = 0
            t_last_yield = time.perf_counter()
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            prompt_token_pad = 0
            while True:
                # if len(self.tts_speech_token_dict[this_uuid]) == 1:
                #     # first token time 
                
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len ##hop_len = 25, prompt_token_pad = 0
                
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                    t_before_t2w = time.perf_counter()
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False,
                                                     profile_timing=profile_timing)
                    token_offset += this_token_hop_len
                    out = {'tts_speech': this_tts_speech.cpu()}
                    if profile_timing:
                        t2w = getattr(self, '_token2wav_timings', {}).get(this_uuid, {})
                        lm_wait_s = t_before_t2w - t_last_yield
                        lm_speed = self.llm_token_speed_metrics.get(this_uuid) or {}
                        out['_profile'] = {
                            'flow_s': t2w.get('flow_s'), 'hift_s': t2w.get('hift_s'),
                            'token2wav_s': t2w.get('token2wav_s'),
                            'lm_wait_s': lm_wait_s,
                            'lm_first_token_s': self.llm_first_token_time_s.get(this_uuid) if token_offset == 0 else None,
                            'lm_first_18_token_s': lm_speed.get('first_18_token_time_s') if token_offset == 0 else None,
                            'lm_every_12_token_s_list': lm_speed.get('every_12_token_time_s_list', []),
                        }
                        t_last_yield = time.perf_counter()
                    yield out
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens
            t_before_t2w = time.perf_counter()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=token_offset,
                                             uuid=this_uuid,
                                             finalize=True,
                                             profile_timing=profile_timing)
            out = {'tts_speech': this_tts_speech.cpu()}
            if profile_timing:
                t2w = getattr(self, '_token2wav_timings', {}).get(this_uuid, {})
                lm_speed = self.llm_token_speed_metrics.get(this_uuid) or {}
                out['_profile'] = {
                    'flow_s': t2w.get('flow_s'), 'hift_s': t2w.get('hift_s'),
                    'token2wav_s': t2w.get('token2wav_s'),
                    'lm_wait_s': t_before_t2w - t_last_yield,
                    'lm_first_token_s': self.llm_first_token_time_s.get(this_uuid),
                    'lm_first_18_token_s': lm_speed.get('first_18_token_time_s'),
                    'lm_every_12_token_s_list': lm_speed.get('every_12_token_time_s_list', []),
                }
            yield out
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.llm_first_token_time_s.pop(this_uuid, None)
            self.llm_token_speed_metrics.pop(this_uuid, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()