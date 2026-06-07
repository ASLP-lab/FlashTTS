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

import logging
import os
import torch
import json
import re
import datetime
import yaml
from typing import Any, Dict
import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR, CosineAnnealing


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == 'torch_ddp':
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    
    # Extract fileid_list_path from lance_dataset_opener config if available
    fileid_list_path = None
    # if 'lance_dataset_opener' in configs:
    #     fileid_list_path = configs['lance_dataset_opener'].get('fileid_list_path', None)
    
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=True, partition=True, fileid_list_path=fileid_list_path)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=False, partition=False, fileid_list_path=fileid_list_path)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs['train_conf']["dtype"] = 'bf16' if args.use_amp is True else 'fp32'
    else:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs['train_conf']["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs['train_conf']["dtype"] = "bf16"
        else:
            configs['train_conf']["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs['train_conf']['save_per_step'] = int(configs['train_conf']['save_per_step'] *
                                                     configs['train_conf']['accum_grad'] / ds_configs["gradient_accumulation_steps"])
        configs['train_conf']['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        configs['train_conf']['grad_clip'] = ds_configs["gradient_clipping"]
        configs['train_conf']['log_interval'] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'CosineAnnealing':
            scheduler_type = CosineAnnealing
            scheduler = CosineAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf']['scheduler'])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":
            def scheduler(opt):
                return scheduler_type(opt, **configs['train_conf']['scheduler_conf'])
            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters())

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.module.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'CosineAnnealing':
            scheduler_type = CosineAnnealing
            scheduler = CosineAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf']['scheduler'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.module.discriminator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save({**model.module.state_dict(), 'epoch': info_dict['epoch'], 'step': info_dict['step']}, save_model_path)
    else:
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=model_name,
                                  client_state=info_dict)
    if rank == 0:
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(rank, save_model_path))


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(group=group_join,
                                   timeout=group_join.options._timeout)
            return False
        except RuntimeError as e:
            logging.info("Detected uneven workload distribution: {}\n".format(e) +
                         "Break current worker to manually join all workers, " +
                         "world_size {}, current rank {}, current local_rank {}\n".
                         format(world_size, rank, local_rank))
            return True
    else:
        return False


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None, speaker_emb_pattern=None, drop_cfg=0.0, use_multi_lingual_token_level=False, kl_weight=1.0):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with autocast:
        info_dict['loss_dict'] = model(batch, device, speaker_emb_pattern=speaker_emb_pattern, drop_cfg=drop_cfg, use_multi_lingual_token_level=use_multi_lingual_token_level)
        if "loss_kl" in info_dict['loss_dict']:
            for i in range(len(info_dict['loss_dict']["loss_kl"])):
                info_dict['loss_dict']["loss_kl"][i] *= kl_weight
                # info_dict['loss_dict']["loss"] += info_dict['loss_dict']["loss_kl"][i]
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict['loss_dict']["chosen_logps"]
            rejected_logps = info_dict['loss_dict']["rejected_logps"]
            sft_loss = info_dict['loss_dict']['loss']
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict['loss_dict']["loss"] = preference_loss + sft_loss
            info_dict['loss_dict']["sft_loss"] = sft_loss
            info_dict['loss_dict']["dpo_loss"] = preference_loss
            info_dict['loss_dict']["dpo_acc"] = dpo_acc
            info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
            info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
    return info_dict


def batch_backward(model, scaler, info_dict):
    """根据当前 loss 执行反向传播。

    说明:
        - 对于 MTP，多头 forward 可能返回 loss list，如 [loss_head0, loss_head1, ...]。
          此处既要保留每个 head 的 loss 统计，又要将其聚合为单标量（默认为平均）用于反向传播。
    """
    loss_entry = info_dict["loss_dict"]["loss"]

    # MTP 情况下，loss 可能是多个 head 的列表: [Tensor, Tensor, ...]
    if isinstance(loss_entry, (list, tuple)):
        # 1) 保留原始各 head loss，便于上层统计/日志使用
        #    形如: [Tensor(head0), Tensor(head1), ...]
        loss_heads = list(loss_entry)
        info_dict.setdefault("loss_dict", {})["loss_heads"] = loss_heads
        
        # 2) 聚合为单个标量 loss（默认做均值），用于实际反向传播
        #    各 head loss 已是标量 Tensor，直接求和 / len 即可
        loss = sum(loss_heads) / float(len(loss_heads))
        if "loss_kl" in info_dict['loss_dict']:
            for i in range(len(info_dict['loss_dict']["loss_kl"])):
                loss += info_dict['loss_dict']["loss_kl"][i]
    else:
        # 非 MTP 情况：单一标量 loss
        loss = loss_entry

    # 统一更新回 info_dict，后续分支复用
    info_dict["loss_dict"]["loss"] = loss

    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(loss)
    else:
        scaled_loss = loss / info_dict["accum_grad"]
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict["loss_dict"]["loss"] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if info_dict['train_engine'] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = int(os.environ.get('RANK', 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (info_dict['train_engine'] == 'deepspeed' and info_dict['is_gradient_accumulation_boundary'] is True) or \
           (info_dict['train_engine'] == 'torch_ddp' and (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0):
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)

            # 展开 MTP 的多 head 指标：loss_heads, acc(list) 等
            flat_loss_dict: Dict[str, Any] = {}
            for k, v in loss_dict.items():
                if isinstance(v, (list, tuple)):
                    # 针对列表类型：逐项展开到 loss_head_i / acc_i 之类的键
                    for idx, item in enumerate(v):
                        key_i = f"{k}_{idx}"
                        flat_loss_dict[key_i] = item
                    # 同时记录一个平均值（例如 acc 或 loss_heads 的 mean）
                    try:
                        mean_val = sum(v) / float(len(v))
                        flat_loss_dict[f"{k}_mean"] = mean_val
                    except Exception:
                        pass
                else:
                    flat_loss_dict[k] = v

            # 写入 tensorboard（只写标量）
            for k, v in flat_loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        # 展开打印，每个 list 型指标也展开为多个项 + mean
        printable_items: Dict[str, float] = {}
        for name, value in loss_dict.items():
            if isinstance(value, (list, tuple)):
                try:
                    mean_val = float((sum(value) / float(len(value))).item())
                    printable_items[f"{name}_mean"] = mean_val
                    for idx, item in enumerate(value):
                        printable_items[f"{name}_{idx}"] = float(item.item())
                except Exception:
                    # 如果无法转换为 float，则跳过展开，直接打印类型信息
                    continue
            else:
                try:
                    printable_items[name] = float(value.item() if hasattr(value, "item") else float(value))
                except Exception:
                    continue

        for name, val in printable_items.items():
            log_str += '{} {:.6f} '.format(name, val)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = int(os.environ.get('RANK', 0))
    logging.info(
        'Epoch {} Step {} CV info lr {} {} rank {}'.format(
            epoch, step + 1, lr, rank, ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)
