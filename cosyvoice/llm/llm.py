# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from transformers import Qwen2Config
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss, KLLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.sos_token='<|start_of_speech|>'
        self.sot_token='<|start_of_text|>'
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1,  1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        # pretrain_path = 'path/to/CosyVoice-BlankEN'
        # self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path, local_files_only=True)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache

class Qwen2LM_MTP(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
            head_num: int = 5,              # Number of training heads (MTP heads)
            inference_head_num: int = 5,    # Number of inference heads (controls speedup ratio)
            mtp_head_num: int = 14,         # Attention heads per MTP layer
            freeze_embedding: bool = False,
            freeze_llm: bool = False,
            use_kl_divergence: bool = False,
            kl_divergence_weight: float = 1.0,
            mtp_top_k: Optional[int] = None,
            eos_top_k: Optional[int] = None,
            temperature: Optional[float] = None,
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.sos_token= torch.tensor(151648)  # '<|start_of_speech|>'
        self.sot_token= torch.tensor(151649)  # '<|start_of_text|>'

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)

        # ===== MTP MODULE CORE =====
        # Multi-head training blocks - the key innovation for parallel token prediction
        self.head_num = head_num
        self.inference_head_num = inference_head_num

        # Create MTP blocks: each is a lightweight transformer decoder layer
        self.mtp_block = nn.ModuleList([
            Qwen2DecoderLayer(
                Qwen2Config(
                    hidden_size=llm_input_size,
                    num_attention_heads=mtp_head_num,
                    num_key_value_heads=mtp_head_num,
                ),
                layer_idx=0,  # All layers use same config
            )
            for _ in range(head_num)
        ])

        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio
        self.spk_embed_convert = torch.nn.Linear(192, llm_input_size)
        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}

        # Freeze embeddings if requested
        if freeze_embedding:
            # Train MTP heads only; freeze speech embedding and decoder
            self.speech_embedding.weight.requires_grad = False
            self.llm_decoder.weight.requires_grad = False

        # Optionally freeze LLM (Qwen2ForCausalLM)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        self.use_kl_divergence = use_kl_divergence
        if self.use_kl_divergence:
            self.kl_divergence_weight = kl_divergence_weight
            logger.info(f"use kl divergence with weight {self.kl_divergence_weight}")
        else:
            self.kl_divergence_weight = 0.0
            logger.info("do not use kl divergence")

        self.mtp_top_k = mtp_top_k if mtp_top_k is not None else 50
        self.eos_top_k = eos_top_k if eos_top_k is not None else 50
        self.temperature = temperature if temperature is not None else 0.7
        self.check_success_count = {
            "mtp_head": 0,
            "total": 0,
            "mtp_head_list": [0] * head_num,
        }


    def forward(
            self,
            batch: dict,
            device: torch.device,
            speaker_emb_pattern: str = 'speech',
            drop_cfg: float = 0.0,
            use_multi_lingual_token_level: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        batch_size = text_token.size(0)

        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        # 1. encode text_token
        if use_multi_lingual_token_level:
            # logging.info(f"use_multi_lingual_token_level: {use_multi_lingual_token_level}")
            text_token_emb = text_token_emb[:, 1:, :]+text_token_emb[:, 0, :].unsqueeze(1).repeat(1, text_token.size(1)-1, 1) # (B, L, D) -> (B, L, D)
        
        sos_emb = self.llm.model.model.embed_tokens(self.sos_token.repeat(batch_size, 1).to(device))
        sot_emb = self.llm.model.model.embed_tokens(self.sot_token.repeat(batch_size, 1).to(device))
        
        if 'spk_embedding' not in batch:
            print("warning: spk_embedding not in batch")
            speaker_embedding = torch.zeros(batch_size, 1, self.llm_input_size, dtype=text_token_emb.dtype, device=device)

        # 2. encode speech_tokens
        speech_token_emb = self.speech_embedding(speech_token)
        lm_target = lm_target.to(device)
        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))

        if speaker_emb_pattern == 'text' or speaker_emb_pattern == 'all':
            # import pdb; pdb.set_trace()
            lm_output = lm_output[:, 2:, :]
        elif speaker_emb_pattern == 'speech':
            # import pdb; pdb.set_trace()
            lm_output = lm_output[:, 1:, :]
        else:
            lm_output = lm_output[:, 1:, :]

        # logits calculation the MTP module
        # Each MTP head processes the LLM output to predict different token positions
        # import pdb; pdb.set_trace()
        mtp_output = [
            self.mtp_block[i](lm_output.transpose(0, 1))[0].transpose(0, 1)
            for i in range(self.head_num)
        ]

        # Each head produces its own logits
        logits = [self.llm_decoder(mtp_output[i]) for i in range(self.head_num)]
        
        losses = [
            self.criterion_ce(logits[i], mtp_target[i].to(device))
            for i in range(self.head_num)
        ]
        
        if self.use_kl_divergence:
            ref_logits = self.llm_decoder(lm_output)  # (B, T, V)
            losses_kl = [
                D.kl_divergence(
                    D.Categorical(logits=logits[i][:, :-i, :] if i > 0 else logits[i]),
                    D.Categorical(logits=ref_logits[:, i:, :] if i > 0 else ref_logits),
                ).mean()
                for i in range(self.head_num)
            ]

        accs = [
            th_accuracy(
                logits[i].view(-1, self.speech_token_size + 3),
                mtp_target[i].to(device),
                ignore_label=IGNORE_ID,
            )
            for i in range(self.head_num)
        ]

        if self.use_kl_divergence:
            losses_kl = [loss * self.kl_divergence_weight for loss in losses_kl]
            return {'loss': losses, 'acc': accs, 'loss_kl': losses_kl}
        else:
            return {'loss': losses, 'acc': accs}

    @torch.inference_mode()
    def inference_with_spkemb(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            spk_embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
            is_from_speech: bool = False,
            is_from_text: bool = False,
            cfg_drop: float = 0.0,
            is_tokenlevel: bool = False,
            is_icl: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        
        text_emb = self.llm.model.model.embed_tokens(text)  # (1, text_len, hidden_size)
        sos_emb = self.llm.model.model.embed_tokens(self.sos_token.to(device)).expand(1, 1, -1)
        sot_emb = self.llm.model.model.embed_tokens(self.sot_token.to(device)).expand(1, 1, -1)

        spk_embedding = F.normalize(spk_embedding, dim=1)
        spk_emb = self.spk_embed_convert(spk_embedding).expand(1, 1, -1)  # (1, 1, hidden_size)
        
        text_emb = text_emb[:, 1:, :]+text_emb[:, 0, :].unsqueeze(1).repeat(1, text.size(1)-1, 1) # (B, L, D) -> (B, L, D)
        
        text_init_emb = torch.cat([sot_emb, text_emb[:,0,:].unsqueeze(0)], dim=1)
        speech_init_emb = torch.concat([spk_emb, sos_emb], dim=1)

        lm_input =  text_init_emb + speech_init_emb # (1, 1+prompt_max_len, hidden_size)
        text_emb_for_inference = text_emb[:,1:,:]
        min_len = int(text_len.item() * min_token_text_ratio)
        max_len = int(text_len.item() * max_token_text_ratio)
        # Step-by-step decode, track first-token time
        start_time = time.perf_counter()
        first_token_hop_len = 18
        every_token_hop_len = 12
        first_18_token_time_s = None
        every_12_token_time_s_list = []
        last_chunk_time = None
        out_token_count = 0
        # inference_wrapper_mtp_fast
        # inference_wrapper_mtp_fast_check
        for token in self.inference_wrapper_mtp_fast_check(lm_input, sampling, min_len, max_len, uuid, text_emb_for_inference):
            # Time every 12 tokens
            out_token_count += 1
            now = time.perf_counter()

            if out_token_count == first_token_hop_len:
                first_18_token_time_s = now - start_time
                last_chunk_time = now
                setattr(self, '_first_token_time_s', first_18_token_time_s)
                setattr(self, '_first_18_token_time_s', first_18_token_time_s)
                setattr(self, '_every_12_token_time_s_list', every_12_token_time_s_list)
                logger.info(f"first 18 tokens time: {first_18_token_time_s:.4f}s")
            elif out_token_count > first_token_hop_len and (out_token_count - first_token_hop_len) % every_token_hop_len == 0:
                if last_chunk_time is not None:
                    chunk_time_s = now - last_chunk_time
                    every_12_token_time_s_list.append(chunk_time_s)
                    setattr(self, '_every_12_token_time_s_list', every_12_token_time_s_list)
                    # logger.info(f"tokens [{out_token_count - every_token_hop_len}-{out_token_count}] (12 tokens) time: {chunk_time_s:.4f}s")
                last_chunk_time = now
            # logger.info(f"token count, check success count: {self.check_success_count}")
            yield token


    @torch.inference_mode()
    def inference_wrapper_mtp_fast_check(self, lm_input, sampling, min_len, max_len, uuid, text_emb_for_inference):
        # ===== MTP Fast Check：NO KV Cache + parallel MTP + delay validation =====
        # Reset per sample for metrics accumulation
        self.check_success_count["mtp_head"] = 0
        self.check_success_count["total"] = 0
        for i in range(len(self.check_success_count["mtp_head_list"])):
            self.check_success_count["mtp_head_list"][i] = 0

        out_tokens: List[int] = []
        head_k = min(self.inference_head_num, self.head_num)
        # head_k = 3
        logger.info(f"head_k: {head_k}")
        # cache = None
        text_len = text_emb_for_inference.shape[1]
        pending_to_validate: List[int] = []  # Tokens to validate this round

        while len(out_tokens) < max_len:
            seq_len = lm_input.shape[1]
            masks = torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool)

            # Step 1: Forward with KV cache
            y_pred, _ = self.llm.forward_one_step(
                lm_input,
                masks=masks,
                cache=None,
            )
            backbone_logits = self.llm_decoder(y_pred)  # (1, seq_len, vocab_size)

            # Step 2: Delayed validation (backbone logits vs previous MTP tokens)
            if len(pending_to_validate) > 0:
                K = len(pending_to_validate)
                validated_ids: List[int] = []
                rollback_at: Optional[int] = None
                for j, tok in enumerate(pending_to_validate):
                    if tok in self.stop_token_ids:
                        validated_ids.append(tok)
                        continue
                    check_pos = seq_len - K + j - 1
                    backbone_logit_j = backbone_logits[0, check_pos, :]
                    if tok in self.stop_token_ids:
                        top_k_tokens = torch.topk(backbone_logit_j, self.eos_top_k, dim=-1).indices
                    else:
                        top_k_tokens = torch.topk(backbone_logit_j, self.mtp_top_k, dim=-1).indices
                    if (top_k_tokens == tok).any().item():
                        validated_ids.append(tok)
                        self.check_success_count["mtp_head"] += 1
                        self.check_success_count["mtp_head_list"][j] += 1
                    else:
                        rollback_at = j
                        break

                if rollback_at is not None:
                    j = rollback_at
                    tok = pending_to_validate[j]
                    check_pos = seq_len - K + j - 1
                    backbone_logit_j = backbone_logits[0, check_pos, :]
                    decoded_snapshot = list(out_tokens)[: -K + j]
                    ignore_eos = (len(decoded_snapshot) + j) < min_len
                    # Resample: logits/T then softmax (T<1 sharper, T>1 smoother)
                    resampled_id = int(self.sampling_ids(
                        (backbone_logit_j / self.temperature).log_softmax(dim=-1),
                        decoded_snapshot,
                        sampling,
                        ignore_eos=ignore_eos
                    ))
                    for t in validated_ids:
                        yield t
                    yield resampled_id
                    num_remove = K - j
                    out_tokens = out_tokens[:-num_remove] + [resampled_id]
                    base_idx = len(out_tokens) - 1
                    group_ids_rollback = [resampled_id]
                    token_embs_rollback = self.speech_embedding.weight[
                        torch.tensor(group_ids_rollback, device=lm_input.device, dtype=torch.long)
                    ].clone()
                    pos_indices_rb = base_idx + torch.arange(len(group_ids_rollback), device=lm_input.device)
                    valid_mask_rb = pos_indices_rb < text_len
                    if valid_mask_rb.any():
                        token_embs_rollback[valid_mask_rb] += text_emb_for_inference[0, pos_indices_rb[valid_mask_rb], :]
                    lm_input = torch.cat([lm_input[:, :-num_remove], token_embs_rollback.unsqueeze(0)], dim=1)
                    pending_to_validate = []
                    if resampled_id in self.stop_token_ids:
                        break
                    continue

                for t in validated_ids:
                    yield t
                    if t in self.stop_token_ids:
                        return
                pending_to_validate = []

            # Step 3: MTP parallel prediction
            last_hidden = y_pred[:, -1, :].unsqueeze(1)
            mtp_outputs = [self.mtp_block[j](last_hidden)[0] for j in range(head_k)]
            logps = [self.llm_decoder(mtp_outputs[j][:, -1]).log_softmax(dim=-1) for j in range(head_k)]

            # Step 4: Sample
            decoded_snapshot = list(out_tokens)
            ignore_eos_flags = [(len(decoded_snapshot) + j) < min_len for j in range(head_k)]
            sampled_ids: List[int] = []
            for j in range(head_k):
                top_id = int(self.sampling_ids(
                    logps[j].squeeze(dim=0),
                    decoded_snapshot,
                    sampling,
                    ignore_eos=ignore_eos_flags[j]
                ))
                sampled_ids.append(top_id)
            self.check_success_count["total"] += head_k

            # Step 5: Stop and state update
            stop = False
            group_ids: List[int] = []
            for top_id in sampled_ids:
                if top_id in self.stop_token_ids:
                    stop = True
                    break
                out_tokens.append(top_id)
                group_ids.append(top_id)
                pending_to_validate.append(top_id)
                if len(out_tokens) >= max_len:
                    stop = True
                    break

            if stop or len(group_ids) == 0:
                if len(pending_to_validate) > 0:
                    for t in pending_to_validate:
                        yield t
                break

            # Step 6: Concat embedding, update lm_input
            num_new_tokens = len(group_ids)
            base_idx = len(out_tokens) - num_new_tokens
            group_ids_tensor = torch.tensor(group_ids, device=lm_input.device, dtype=torch.long)
            token_embs = self.speech_embedding.weight[group_ids_tensor].clone()
            pos_indices = base_idx + torch.arange(num_new_tokens, device=lm_input.device)
            valid_mask = pos_indices < text_len
            if valid_mask.any():
                token_embs[valid_mask] = token_embs[valid_mask] + text_emb_for_inference[0, pos_indices[valid_mask], :]
            lm_input = torch.cat([lm_input, token_embs.unsqueeze(0)], dim=1)
            

    @torch.inference_mode()
    def inference_wrapper_mtp(self, lm_input, sampling, min_len, max_len, uuid, text_emb_for_inference):
        # ===== STANDARD MTP INFERENCE PATH =====
        out_tokens: List[int] = []
        head_k = min(self.inference_head_num, self.head_num)
        logger.info(f"head_k: {head_k}")
        # head_k = 5

        while len(out_tokens) < max_len:
            # Step 1: Run base LLM once on current prefix
            y_pred, _ = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(torch.ones(
                    (1, lm_input.shape[1], lm_input.shape[1]),
                    device=lm_input.device
                )).to(torch.bool),
                cache=None,  # No caching for simplicity
            )

            # Step 2: Use MTP heads to predict next K tokens in parallel
            last_hidden = y_pred[:, -1, :].unsqueeze(1)  # (B, 1, H)
            mtp_outputs = [
                self.mtp_block[j](last_hidden)[0]#.transpose(0, 1) # (head_k, B, 1, H) -> (B, head_k, 1, H)
                for j in range(head_k)
            ]

            # Get logits for each head
            logps = [
                self.llm_decoder(mtp_outputs[j][:, -1]).log_softmax(dim=-1)
                for j in range(head_k)
            ]

            # Step 3: Sample tokens from each head
            decoded_snapshot = list(out_tokens)  # Current sequence snapshot
            sampled_ids: List[int] = []

            for j in range(head_k):
                ignore_eos = (len(decoded_snapshot) + j) < min_len
                top_id = self.sampling_ids(
                    logps[j].squeeze(dim=0),
                    decoded_snapshot,
                    sampling,
                    ignore_eos=ignore_eos
                )
                sampled_ids.append(int(top_id))

            # Step 4: Yield sampled tokens and update sequence
            group_ids: List[int] = []
            stop = False

            for top_id in sampled_ids:
                if top_id in self.stop_token_ids:
                    stop = True
                    break
                yield top_id
                out_tokens.append(top_id)
                group_ids.append(top_id)

                if len(out_tokens) >= max_len:
                    stop = True
                    break

            if stop or len(group_ids) == 0:
                break

            # Step 5: Append all predicted token embeddings to input (speech_emb + text_emb as in inference_wrapper_cfg)
            base_idx = len(out_tokens) - len(group_ids)
            text_len = text_emb_for_inference.shape[1]
            token_embs = self.speech_embedding.weight[
                torch.tensor(group_ids, device=lm_input.device, dtype=torch.long)
            ].clone()  # (K, H)
            pos_indices = base_idx + torch.arange(len(group_ids), device=lm_input.device, dtype=torch.long)
            valid_mask = pos_indices < text_len
            if valid_mask.any():
                token_embs[valid_mask] = token_embs[valid_mask] + text_emb_for_inference[0, pos_indices[valid_mask], :].to(lm_input.device)
            token_embs = token_embs.unsqueeze(0)  # (1, K, H)

            lm_input = torch.cat([lm_input, token_embs], dim=1)


    @torch.inference_mode()
    def inference_wrapper_mtp_fast(self, lm_input, sampling, min_len, max_len, uuid, text_emb_for_inference):
        out_tokens: List[int] = []
        head_k = min(self.inference_head_num, self.head_num)
        logger.info(f"head_k: {head_k}")

        # Initialize KV cache for base LLM
        # cache = None
        text_len = text_emb_for_inference.shape[1]

        while len(out_tokens) < max_len:
            # Step 1: Run base LLM once on current prefix with caching
            seq_len = lm_input.shape[1]
            masks = torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool)

            y_pred, _ = self.llm.forward_one_step(
                lm_input,
                masks=masks,
                cache=None,  # Use KV cache for acceleration
            )

            # Step 2: Fully parallelized MTP heads processing
            last_hidden = y_pred[:, -1, :].unsqueeze(1)  # (B, 1, H)

            # Process all MTP heads in parallel using efficient batching
            # Each head gets the same input but has different learned parameters
            mtp_outputs = []

            mtp_results = [self.mtp_block[j](last_hidden) for j in range(head_k)]
            mtp_outputs_parallel = [result[0] for result in mtp_results]
            mtp_outputs = torch.stack(mtp_outputs_parallel, dim=0)

            # Get logits for all heads in parallel
            # mtp_outputs: (head_k, B, 1, H) -> (head_k, B, H) -> (head_k, B, vocab_size)
            logits_flat = self.llm_decoder(mtp_outputs.squeeze(2))  # (head_k, B, vocab_size)
            logps = logits_flat.log_softmax(dim=-1)  # (head_k, B, vocab_size)

            # Step 3: Optimized sampling with pre-computed flags
            decoded_snapshot = list(out_tokens)
            base_seq_len = len(decoded_snapshot)

            # Pre-compute ignore_eos flags for all heads
            ignore_eos_flags = [(base_seq_len + j) < min_len for j in range(head_k)]

            # Sample tokens from each head (sequential due to potential EOS retry logic)
            sampled_ids = []
            for j in range(head_k):
                logp_j = logps[j, 0]  # (vocab_size,)
                top_id = self.sampling_ids(
                    logp_j,
                    decoded_snapshot,
                    sampling,
                    ignore_eos=ignore_eos_flags[j]
                )
                sampled_ids.append(int(top_id))

            # Step 4: Yield sampled tokens and update sequence
            group_ids: List[int] = []
            stop = False

            for top_id in sampled_ids:
                if top_id in self.stop_token_ids:
                    stop = True
                    break
                yield top_id
                out_tokens.append(top_id)
                group_ids.append(top_id)

                if len(out_tokens) >= max_len:
                    stop = True
                    break

            if stop or len(group_ids) == 0:
                break

            # Step 5: Memory-efficient embedding concatenation
            num_new_tokens = len(group_ids)
            base_idx = len(out_tokens) - num_new_tokens

            # Create token indices tensor once
            group_ids_tensor = torch.tensor(group_ids, device=lm_input.device, dtype=torch.long)

            # Get speech embeddings without unnecessary clone
            token_embs = self.speech_embedding.weight[group_ids_tensor]  # (K, H)

            # Vectorized text embedding fusion with in-place operations where possible
            if base_idx < text_len:
                pos_indices = base_idx + torch.arange(num_new_tokens, device=lm_input.device, dtype=torch.long)
                valid_mask = pos_indices < text_len
                if valid_mask.any():
                    # In-place addition for memory efficiency
                    token_embs[valid_mask] += text_emb_for_inference[0, pos_indices[valid_mask], :]

            # Expand to batch dimension and concatenate
            token_embs = token_embs.unsqueeze(0)  # (1, K, H)
            lm_input = torch.cat([lm_input, token_embs], dim=1)


    @torch.inference_mode()
    def inference_wrapper_cfg(self, lm_input, lm_input_uncond, sampling, min_len, max_len, uuid, text, text_ucond, cfg_drop):
        pad_emb = self.llm_embedding.weight[0].reshape(1, 1, -1)
        pad_emb = torch.zeros(1, 1, 896, dtype=pad_emb.dtype).to(lm_input.device)
        if hasattr(self, 'vllm'):
            from vllm import SamplingParams, RequestOutput
            sampling_params = SamplingParams(top_k=sampling,
                                             stop_token_ids=self.stop_token_ids,
                                             min_tokens=min_len,
                                             max_tokens=max_len)
            
            with self.lock:
                self.vllm.add_request(uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)}, sampling_params)
                self.vllm_output_queue[uuid] = queue.Queue()
            out_tokens = []
            while True:
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(top_ids)
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    # in stream mode, yield token one by one
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == max_len:
                        break
                time.sleep(0.001)
            with self.lock:
                self.vllm_output_queue.pop(uuid)
        else:
            # ===== MTP + CFG INFERENCE PATH (align with Qwen2LM inference_wrapper_cfg) =====
            # Use cache=None so mask length always equals input length (avoids 4 vs 6 mismatch
            # when past_key_values is used with full-sequence input in HuggingFace).
            out_tokens: List[int] = []
            head_k = min(self.inference_head_num, self.head_num)
            text_pos: int = 0

            while len(out_tokens) < max_len:
                # Step 1: Run base LLM once on current prefix (cond / uncond)
                y_pred, _ = self.llm.forward_one_step(
                    lm_input,
                    masks=torch.tril(
                        torch.ones(
                            (1, lm_input.shape[1], lm_input.shape[1]),
                            device=lm_input.device,
                        )
                    ).to(torch.bool),
                    cache=None,
                )
                y_pred_uncond, _ = self.llm.forward_one_step(
                    lm_input_uncond,
                    masks=torch.tril(
                        torch.ones(
                            (1, lm_input_uncond.shape[1], lm_input_uncond.shape[1]),
                            device=lm_input_uncond.device,
                        )
                    ).to(torch.bool),
                    cache=None,
                )

                # Step 2: Use MTP heads to predict next K tokens in parallel (cond / uncond)
                last_hidden = y_pred[:, -1, :].unsqueeze(1)          # (B, 1, H)
                last_hidden_uncond = y_pred_uncond[:, -1, :].unsqueeze(1)

                mtp_out = [self.mtp_block[j](last_hidden)[0] for j in range(head_k)]
                mtp_out_uncond = [self.mtp_block[j](last_hidden_uncond)[0] for j in range(head_k)]

                # Step 3: Decoder + CFG combine for each head
                logps = [
                    self.llm_decoder(mtp_out[j][:, -1]).log_softmax(dim=-1)
                    for j in range(head_k)
                ]
                logps_uncond = [
                    self.llm_decoder(mtp_out_uncond[j][:, -1]).log_softmax(dim=-1)
                    for j in range(head_k)
                ]
                cfg_logits_list = [
                    logps_uncond[j] + cfg_drop * (logps[j] - logps_uncond[j])
                    for j in range(head_k)
                ]

                # Step 4: Sample tokens from each head (multi-token prediction)
                decoded_snapshot = list(out_tokens)
                sampled_ids: List[int] = []
                for j in range(head_k):
                    ignore_eos = (len(decoded_snapshot) + j) < min_len
                    top_id = self.sampling_ids(
                        cfg_logits_list[j].squeeze(dim=0),
                        decoded_snapshot,
                        sampling,
                        ignore_eos=ignore_eos,
                    )
                    sampled_ids.append(int(top_id))

                # Step 5: Yield sampled tokens and update sequences
                group_ids: List[int] = []
                stop = False
                for top_id in sampled_ids:
                    if top_id in self.stop_token_ids:
                        stop = True
                        break
                    if top_id > self.speech_token_size:
                        continue
                    yield top_id
                    out_tokens.append(top_id)
                    group_ids.append(top_id)

                    if len(out_tokens) >= max_len:
                        stop = True
                        break

                if stop or len(group_ids) == 0:
                    break

                # Step 6: Fuse speech + aligned text embedding per new token
                speech_embs = self.speech_embedding.weight[
                    torch.tensor(group_ids, device=lm_input.device, dtype=torch.long)
                ].unsqueeze(0)  # (1, G, H)

                fused_cond_list = []
                fused_uncond_list = []
                for j in range(speech_embs.size(1)):
                    this_speech = speech_embs[:, j : j + 1, :]  # (1, 1, H)
                    if text_pos < int(text.shape[1]):
                        text_cond = text[:, text_pos, :].unsqueeze(1)        # (1, 1, H)
                        text_u = text_ucond[:, text_pos, :].unsqueeze(1)     # (1, 1, H)
                        fused_cond_list.append(this_speech + text_cond)
                        fused_uncond_list.append(this_speech + text_u)
                        text_pos += 1
                    else:
                        fused_cond_list.append(this_speech)
                        fused_uncond_list.append(this_speech)

                new_token_embs_cond = torch.cat(fused_cond_list, dim=1)      # (1, G, H)
                new_token_embs_uncond = torch.cat(fused_uncond_list, dim=1)  # (1, G, H)

                lm_input = torch.cat([lm_input, new_token_embs_cond], dim=1)
                lm_input_uncond = torch.cat([lm_input_uncond, new_token_embs_uncond], dim=1)


    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid, text):
        pad_emb = self.llm_embedding.weight[0].reshape(1, 1, -1)
        pad_emb = torch.zeros(1, 1, 896, dtype=pad_emb.dtype).to(lm_input.device)
        if hasattr(self, 'vllm'):
            from vllm import SamplingParams, RequestOutput
            sampling_params = SamplingParams(top_k=sampling,
                                             stop_token_ids=self.stop_token_ids,
                                             min_tokens=min_len,
                                             max_tokens=max_len)
            
            with self.lock:
                self.vllm.add_request(uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)}, sampling_params)
                self.vllm_output_queue[uuid] = queue.Queue()
            out_tokens = []
            while True:
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(top_ids)
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    # in stream mode, yield token one by one
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == max_len:
                        break
                time.sleep(0.001)
            with self.lock:
                self.vllm_output_queue.pop(uuid)
        else:
            out_tokens = []
            cache = None
            # gt_tokens = [3701, 4218, 4137, 4137, 4137, 4218, 4137, 4137, 1965, 4068, 2459, 4700, 218, 1526, 5238, 3720, 3155, 3146, 1517, 1951, 1951, 2031, 4218, 4218, 4218, 1923, 2347, 5021, 4913, 4993, 4830, 5260, 6381, 6454, 3556, 317, 1851, 3219, 5247, 5258, 4581, 2637, 1284, 4646, 1220, 1250, 2059, 3403, 3645, 3753, 1539, 1704, 6021, 6048, 6051, 144, 6077, 6320, 6157, 3137, 887, 872, 6074, 6073, 6048, 5133, 2004, 5403, 272, 2148, 2158, 1393, 3570, 1063, 2267, 5238, 3135, 3016, 5463, 6486, 6405, 6405, 6405, 6405, 6082]
            # from tqdm import tqdm
            for i in range(max_len): #tqdm(range(max_len), desc='Decoding'):
                y_pred, cache = self.llm.forward_one_step(lm_input,
                                                          masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                          cache=cache)
                
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
                if top_ids == self.speech_token_size: #6561
                    break
                if top_ids > self.speech_token_size:
                    print(f'top_ids: {top_ids} {i}')
                    # print(f'gt_tokens[i]: {gt_tokens[i]}')
                    continue
                # in stream mode, yield token one by one
                yield top_ids
                out_tokens.append(top_ids)

                # top_ids = gt_tokens[i] if i < len(gt_tokens) else top_ids
                # if i == int(text.shape[1])-1:
                #     top_ids = self.speech_token_size + 2
                #     break

                if i < int(text.shape[1]):
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)+text[:,i,:].unsqueeze(0)
                else:
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)