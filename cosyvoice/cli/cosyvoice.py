#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import re
import glob
from typing import Generator, Optional, Tuple
from tqdm import tqdm
from loguru import logger
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, FlashTTS
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


def find_checkpoint_by_step(model_dir, target_step):
    """Find checkpoint file matching target_step; returns None if not found."""
    pattern = os.path.join(model_dir, 'epoch_*_step_*.pt')
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    for ckpt_file in checkpoint_files:
        match = re.search(r'epoch_(\d+)_step_(\d+)\.pt', ckpt_file)
        if match:
            step = int(match.group(2))
            if step == target_step:
                return ckpt_file
    
    return None


def find_latest_checkpoint(model_dir, min_step=70000):
    """Find latest epoch_X_step_XXXXX.pt with step >= min_step; sort by epoch then step desc."""
    pattern = os.path.join(model_dir, 'epoch_*_step_*.pt')
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    valid_checkpoints = []
    for ckpt_file in checkpoint_files:
        match = re.search(r'epoch_(\d+)_step_(\d+)\.pt', ckpt_file)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            if min_step is None or step >= min_step:
                valid_checkpoints.append((epoch, step, ckpt_file))
    
    if not valid_checkpoints:
        return None
    valid_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid_checkpoints[0][2]


def infer_mtp_yaml_from_model_dir(model_dir):
    """Infer MTP yaml suffix (e.g. mtp2, mtp7) from model_dir path; match longer first."""
    if not model_dir:
        return None
    for suffix in ('mtp10', 'mtp7','mtp5', 'mtp3', 'mtp2'):
        if suffix in model_dir:
            return suffix
    return None


def find_yaml_by_step(model_dir, target_step):
    """Find yaml file matching target_step; returns None if not found."""
    pattern = os.path.join(model_dir, 'epoch_*_step_*.yaml')
    yaml_files = glob.glob(pattern)
    
    if not yaml_files:
        return None
    for yaml_file in yaml_files:
        match = re.search(r'epoch_(\d+)_step_(\d+)\.yaml', yaml_file)
        if match:
            step = int(match.group(2))
            if step == target_step:
                return yaml_file
    return None


def find_latest_yaml(model_dir, min_step=None):
    """Find latest epoch_X_step_XXXXX.yaml with step >= min_step."""
    pattern = os.path.join(model_dir, 'epoch_*_step_*.yaml')
    yaml_files = glob.glob(pattern)
    
    if not yaml_files:
        return None
    valid_yamls = []
    for yaml_file in yaml_files:
        match = re.search(r'epoch_(\d+)_step_(\d+)\.yaml', yaml_file)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            if min_step is None or step >= min_step:
                valid_yamls.append((epoch, step, yaml_file))
    
    if not valid_yamls:
        return None
    valid_yamls.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return valid_yamls[0][2]


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True, is_from_speech=False, is_from_text=False, cfg_drop=0.0, is_tokenlevel=False, is_icl=False, collect_metrics=False):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend), disable=collect_metrics):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id, is_from_speech, is_from_text, cfg_drop)
            req_start = time.perf_counter()
            prev_chunk_time = req_start
            if not collect_metrics:
                logging.info('synthesis text {}'.format(i))
            chunk_idx = 0
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed, cfg_drop=cfg_drop, is_tokenlevel=is_tokenlevel, is_icl=is_icl, profile_timing=(collect_metrics and stream)):
                chunk_time = time.perf_counter()
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                if collect_metrics:
                    m = {
                        'chunk_idx': chunk_idx,
                        'chunk_audio_s': speech_len,
                        'chunk_latency_s': chunk_time - prev_chunk_time,
                        'ttfb_s': chunk_time - req_start if chunk_idx == 0 else None,
                    }
                    if '_profile' in model_output:
                        m.update(model_output['_profile'])
                        del model_output['_profile']
                    model_output['_stream_metrics'] = m
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (chunk_time - req_start) / speech_len))
                elif not collect_metrics:
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (chunk_time - req_start) / speech_len))
                yield model_output
                prev_chunk_time = chunk_time
                chunk_idx += 1

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class FlashTTS(CosyVoice):
    
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1, step=None, pretrained_model_dir="./pretrained_models/FlashTTS-0.5B", mtp_yaml=None, mtp_top_k=None, eos_top_k=None, temperature=None, inference_head_num=None):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        logger.info(f"pretrained_model_dir: {pretrained_model_dir}")
        logger.info(f"model_dir: {model_dir}")
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        
        # MTP yaml: explicit mtp_yaml, else infer from model_dir, else default order
        hyper_yaml_path = None
        mtp_suffix = mtp_yaml
        if mtp_suffix is None:
            mtp_suffix = infer_mtp_yaml_from_model_dir(model_dir)
        if mtp_suffix:
            candidate = '{}/cosyvoice2_{}.yaml'.format(pretrained_model_dir, mtp_suffix)
            if os.path.exists(candidate):
                hyper_yaml_path = candidate
                logger.info(f"Using MTP yaml (explicit or inferred from model_dir): {os.path.basename(hyper_yaml_path)}")
        # Default order when no MTP specified or inferred
        if hyper_yaml_path is None or not os.path.exists(hyper_yaml_path):
            default_yaml = '{}/cosyvoice2.yaml'.format(pretrained_model_dir)
            default_yaml_mtp3 = '{}/cosyvoice2_mtp3.yaml'.format(pretrained_model_dir)
            default_yaml_mtp5 = '{}/cosyvoice2_mtp5.yaml'.format(pretrained_model_dir)
            default_yaml_mtp7 = '{}/cosyvoice2_mtp7.yaml'.format(pretrained_model_dir)
            default_yaml_mtp10 = '{}/cosyvoice2_mtp10.yaml'.format(pretrained_model_dir)
            if os.path.exists(default_yaml_mtp3):
                hyper_yaml_path = default_yaml_mtp3
                logger.info(f"No step-specific yaml found, using default: {os.path.basename(hyper_yaml_path)}")
            elif os.path.exists(default_yaml_mtp5):
                hyper_yaml_path = default_yaml_mtp5
                logger.info(f"No step-specific yaml found, using default: {os.path.basename(hyper_yaml_path)}")
            elif os.path.exists(default_yaml_mtp7):
                hyper_yaml_path = default_yaml_mtp7
                logger.info(f"No step-specific yaml found, using default: {os.path.basename(hyper_yaml_path)}")
            elif os.path.exists(default_yaml_mtp10):
                hyper_yaml_path = default_yaml_mtp10
                logger.info(f"No step-specific yaml found, using default: {os.path.basename(hyper_yaml_path)}")
            elif os.path.exists(default_yaml):
                hyper_yaml_path = default_yaml
                logger.info(f"No step-specific yaml found, using default: {os.path.basename(hyper_yaml_path)}")
            else:
                raise ValueError(f'Yaml file not found! Tried step-specific yaml and default: {default_yaml}')
        
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(pretrained_model_dir, 'CosyVoice-BlankEN')})
        # assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(pretrained_model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(pretrained_model_dir),
                                          '{}/spk2info.pt'.format(pretrained_model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')

        model_cls = get_model_type(configs)
        use_flashtts = model_cls is FlashTTS

        if use_flashtts:
            try:
                from jit_meanflow_xpred.infer.infer_meanflow_jit_xpred import initialize_model, initialize_vocoder
            except ImportError:
                raise ImportError("MeanFlow backend requires jit_meanflow_xpred. Install or add repo to path.")
            _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            _meanflow_config = os.path.join(pretrained_model_dir, "meanflow_config.yaml")
            if not os.path.exists(_meanflow_config):
                _meanflow_config = os.path.join(_root, "jit_meanflow_xpred", "configs", "fm_10ms_contrasive_ecapa_pmeanflow.yaml")
            _meanflow_ckpt = os.path.join(pretrained_model_dir, "meanflow.pt")
            if not os.path.exists(_meanflow_ckpt):
                _meanflow_ckpt = os.path.join(pretrained_model_dir, "meanflow", "model_200000.pt")
            _vocoder_config = os.path.join(pretrained_model_dir, "vocoder_config.json")
            _vocoder_ckpt = os.path.join(pretrained_model_dir, "vocoder.pt")
            if not os.path.exists(_vocoder_config):
                _vocoder_config = os.path.join(_root, "third_party", "hifi-gan", "config_streamfm10ms.json")
            if not os.path.exists(_vocoder_ckpt):
                _vocoder_ckpt = os.path.join(pretrained_model_dir, "vocoder", "g_00400000")
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            meanflow_model = initialize_model(_meanflow_config, _meanflow_ckpt, _device)
            vocoder = initialize_vocoder(_vocoder_config, _vocoder_ckpt, _device)
            self.model = FlashTTS(configs["llm"], meanflow_model, vocoder, fp16)
        else:
            self.model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16)

        if hasattr(self.model, "llm"):
            if mtp_top_k is not None and hasattr(self.model.llm, "mtp_top_k"):
                self.model.llm.mtp_top_k = mtp_top_k
                logger.info(f"MTP override: mtp_top_k={mtp_top_k}")
            if eos_top_k is not None and hasattr(self.model.llm, "eos_top_k"):
                self.model.llm.eos_top_k = eos_top_k
                logger.info(f"MTP override: eos_top_k={eos_top_k}")
            if temperature is not None and hasattr(self.model.llm, "temperature"):
                self.model.llm.temperature = temperature
                logger.info(f"MTP override: temperature={temperature}")
            if inference_head_num is not None and hasattr(self.model.llm, "inference_head_num"):
                self.model.llm.inference_head_num = min(int(inference_head_num), self.model.llm.head_num)
                logger.info(f"MTP override: inference_head_num={self.model.llm.inference_head_num}")

        logger.info(f"model_dir_steps: {step}")
        if step is not None:
            llm_checkpoint = find_checkpoint_by_step(model_dir, step)
            if llm_checkpoint is None:
                logging.warning("Checkpoint with step {} not found, falling back to latest checkpoint".format(step))
                llm_checkpoint = find_latest_checkpoint(model_dir, min_step=70000)
            if llm_checkpoint is None:
                llm_checkpoint = "{}/llm.pt".format(model_dir)
                logging.info("No checkpoint found, using default llm.pt")
            else:
                logging.info("Found checkpoint: {}".format(os.path.basename(llm_checkpoint)))
        else:
            llm_checkpoint = find_latest_checkpoint(model_dir, min_step=50000)
            if llm_checkpoint is None:
                llm_checkpoint = "{}/llm.pt".format(model_dir)
                logging.info("No epoch_*_step_*.pt found, using default llm.pt")
            else:
                logging.info("Found latest checkpoint: {}".format(os.path.basename(llm_checkpoint)))

        if use_flashtts:
            self.model.load(llm_checkpoint)
        else:
            self.model.load(llm_checkpoint,
                            "{}/flow.pt".format(pretrained_model_dir),
                            "{}/hift.pt".format(pretrained_model_dir))
        if load_vllm:
            self.model.load_vllm("{}/vllm".format(pretrained_model_dir))
        if not use_flashtts and load_jit:
            self.model.load_jit("{}/flow.encoder.{}.zip".format(pretrained_model_dir, "fp16" if self.fp16 else "fp32"))
        if not use_flashtts and load_trt:
            self.model.load_trt(
                "{}/flow.decoder.estimator.{}.mygpu.plan".format(pretrained_model_dir, "fp16" if self.fp16 else "fp32"),
                "{}/flow.decoder.estimator.fp32.onnx".format(pretrained_model_dir),
                trt_concurrent,
                self.fp16,
            )
        del configs


# Backward compatibility: alias for FlashTTS
CosyVoice2 = FlashTTS