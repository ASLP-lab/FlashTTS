import torch

from third_party.campplus.fbanks import batch_fbank
from third_party.campplus.DTDNN import CAMPPlus

import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.yml", local_model_path=None, local_config_path=None):
    """
    加载模型文件，优先使用本地路径，如果没有提供本地路径则从 HuggingFace 下载
    
    Args:
        repo_id: HuggingFace 仓库ID
        model_filename: 模型文件名
        config_filename: 配置文件文件名
        local_model_path: 本地模型文件路径，如果提供则直接使用
        local_config_path: 本地配置文件路径，如果提供则直接使用
        
    Returns:
        如果 config_filename 为 None，返回模型路径；否则返回 (模型路径, 配置文件路径)
    """
    # 如果提供了本地模型路径，直接使用
    if local_model_path is not None and os.path.exists(local_model_path):
        if config_filename is None:
            return local_model_path
        # 如果提供了本地配置文件路径，直接使用
        if local_config_path is not None and os.path.exists(local_config_path):
            return local_model_path, local_config_path
        # 否则尝试从 HuggingFace 下载配置文件
        ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(ckpt_path, exist_ok=True)
        config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=ckpt_path)
        return local_model_path, config_path
    
    # 如果没有提供本地路径，从 HuggingFace 下载
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=ckpt_path)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=ckpt_path)

    return model_path, config_path


class SpkEmbedding:
    def __init__(self, device, half=False, model_path=None):
        """
        初始化 SpkEmbedding
        
        Args:
            device: 设备 ('cpu' 或 'cuda')
            half: 是否使用半精度
            model_path: 本地模型文件路径，如果提供则直接使用，否则从 HuggingFace 下载
        """
        self.device = device
        self.half = half
        self.encoder = CAMPPlus(feat_dim=80, embedding_size=192)
        # 如果提供了本地模型路径，直接使用；否则从 HuggingFace 下载
        campplus_sd_path = load_custom_model_from_hf(
            "funasr/campplus", 
            "campplus_cn_common.bin", 
            config_filename=None,
            local_model_path=model_path
        )
        campplus_sd = torch.load(campplus_sd_path, map_location='cpu')
        self.encoder.load_state_dict(campplus_sd)
        self.encoder.eval()
        if self.half:
            self.encoder = self.encoder.half()
        self.encoder.to(device)


    def extract_features(self, wavs, wav_lengths, max_duration=5.0):
        with torch.no_grad():  # 确保不需要梯度
            waves = [
                (wave[:min(wave_length, int(max_duration * 16000))] * (1 << 15)).type(torch.float32) for wave, wave_length in zip(wavs, wav_lengths)
            ]

            feats = batch_fbank(
                waves,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat_lengths = torch.tensor([feat.shape[0] for feat in feats])
            feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
            feats = feats - feats.mean(dim=1, keepdim=True)
            if self.half:
                feats = feats.half()
            spkembeds = [self.encoder(feat.unsqueeze(0)) for feat in feats]
            spkembeds = torch.cat(spkembeds, dim=0)
            if self.half:
                spkembeds = spkembeds.float()
            # 确保返回的tensor可以序列化：detach并移到CPU
            spkembeds = spkembeds.detach().cpu()
        return spkembeds
