# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
# print_layer_time = True
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import threading
import copy
import torch.nn.functional as F
print_layer_time = False
if print_layer_time:
    import time
loading_computation_parallel = True

class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 use_sliding_window: bool = False,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window if use_sliding_window else None

        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=self.sliding_window)
        self.hack_kv = []
        self.attention = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,

        status,
        cache_fuse_metadata,
        old_kv,
    ) -> torch.Tensor:
        # if status in [1,2]:
        #     import pdb; pdb.set_trace()
        qkv, _ = self.qkv_proj(hidden_states)
        
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # if status in [1,2]:
        #     # import pdb; pdb.set_trace()
        #     if cache_fuse_metadata["fake_q"] is None:
        #         cache_fuse_metadata['fake_q'] = torch.rand_like(q)
        #     try:
        #         _, old_kv[0] = self.rotary_emb(cache_fuse_metadata['org_pos'],
        #                                     cache_fuse_metadata['fake_q'],
        #                                     old_kv[0])
        #     except:
        #         import pdb; pdb.set_trace()
            
        if cache_fuse_metadata['collect']:
            self.hack_kv = [k.clone(), v.clone()]
        # if status in [1,2]:
        #     import pdb; pdb.set_trace()
        # q, k = self.rotary_emb(positions, q, k)
        # if k.shape[0]==578:
        #     import pdb; pdb.set_trace()


        q, k = self.rotary_emb(positions, q, k)
        # try:
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                status, cache_fuse_metadata, old_kv,
                                self.kv_scale)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        use_sliding_window = (config.use_sliding_window
                              and layer_idx < config.max_window_layers)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            linear_method=linear_method,
            sliding_window=config.sliding_window)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],

        status: int,
        cache_fuse_metadata: dict,
        old_kv,
        layer_id=None,
        kv_schedule=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        
        if loading_computation_parallel and cache_fuse_metadata["check"]:
            if print_layer_time:
                start = time.time()
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
            
            if status == 1:
                residual = residual[cache_fuse_metadata["query_imp_indices"]]
                hidden_states = hidden_states[cache_fuse_metadata["query_imp_indices"]]
                
            old_kv_next_layer = None

            def calculate_recomputation_modules_next_layer():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    nonlocal cache_fuse_metadata
                    # output.view(-1, self.num_heads * self.head_size)
                    attention = self.self_attn.attn.impl.attention_weights
                    self.calculate_recomputation_modules(layer_id+1,attention,cache_fuse_metadata)
                    if print_layer_time:
                        print('calculate_recomputation_modules_next_layer',time.time()-start)
            
            def load_kv_next_layer():
            ##!!!!!!!
                with torch.no_grad():
                    nonlocal old_kv_next_layer
                    old_kv_next_layer = kv_schedule.move_kv_layer_wise_vllm_v1(layer_id+1,cache_fuse_metadata["recompute_modules"])

            def load_kv_extra_layer():
            ##!!!!!!!!!
                with torch.no_grad():
                    nonlocal cache_fuse_metadata
                    kv_schedule.move_kv_layer_wise_extra_vllm_v1(layer_id+2,cache_fuse_metadata)

            def attention():
            ##!!!!!!!!!!!
                with torch.no_grad():
                    nonlocal hidden_states
                    # if print_layer_time:
                    #     start = time.time()
                    hidden_states = self.self_attn(
                        positions=positions,
                        hidden_states=hidden_states,
                        kv_cache=kv_cache,
                        attn_metadata=attn_metadata,

                        status=status,
                        cache_fuse_metadata=cache_fuse_metadata,
                        old_kv=old_kv,
                    )
                    # if print_layer_time:
                    #     print(f'attention_with_loading_time: {time.time()-start}')
                    #     start = time.time()

            def mlp():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    nonlocal hidden_states, residual
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual)
                    hidden_states = self.mlp(hidden_states)
                    if print_layer_time:
                        print(f'real mlp time: {time.time()-start}')

            def no_check_attn_mlp():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    attention()
                    if print_layer_time:
                        print(f'attention_time: {time.time()-start}')
                        start = time.time()
                    mlp()
                    if print_layer_time:
                        print(f'mlp_time: {time.time()-start}')

            def no_check_load_kv():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    load_kv_next_layer()
                    if print_layer_time:
                        print(f'load_kv_next_layer: {time.time()-start}')
                        start = time.time()
                    while thread2.is_alive() and cache_fuse_metadata["finish_load"]==False:
                        # print('????')
                        load_kv_extra_layer()
                    # print('over')
                    if print_layer_time:
                        print(f'loading_kv_extra_time: {time.time()-start}')
            
            def check_load_kv():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    load_kv_next_layer()
                    while thread2.is_alive() and cache_fuse_metadata["finish_load"]==False:
                        # print('!!!!')
                        load_kv_extra_layer()
                    # print('over')
                    if print_layer_time:
                        print(f'loading_kv1_time: {time.time()-start}')

            
            def check_load_kv2():
                with torch.no_grad():
                    if print_layer_time:
                        start = time.time()
                    while (thread3.is_alive() or thread4.is_alive()) and cache_fuse_metadata["finish_load"]==False:
                        load_kv_extra_layer()
                    if print_layer_time:
                        print(f'loading_kv2_time: {time.time()-start}')

            if print_layer_time:
                start = time.time()
            if layer_id+1 in cache_fuse_metadata["check_layers"]:
                if status == 1:
                    status = 3
                else:
                    status = 4

                thread2 = threading.Thread(target=attention)
                thread1 = threading.Thread(target=check_load_kv)

                thread2.start()
                thread1.start()
            
                thread2.join()
                thread1.join()
                if print_layer_time:
                    print(f'attention_with_loading_time: {time.time()-start}')
                    start = time.time()

                thread4 = threading.Thread(target=mlp)
                thread3 = threading.Thread(target=calculate_recomputation_modules_next_layer)
                thread5 = threading.Thread(target=check_load_kv2)

                thread4.start()
                thread3.start()
                thread5.start()
            
                thread4.join()
                thread3.join()
                thread5.join()
                if print_layer_time:
                    print(f'mlp_time: {time.time()-start}')
            else:
                thread2 = threading.Thread(target=no_check_attn_mlp)
                thread1 = threading.Thread(target=no_check_load_kv)

                thread2.start()
                thread1.start()
            
                thread2.join()
                thread1.join()

                


        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
                
            if status == 1:
                residual = residual[cache_fuse_metadata["query_imp_indices"]]
                hidden_states = hidden_states[cache_fuse_metadata["query_imp_indices"]]

            if cache_fuse_metadata["check"]:
                if layer_id+1 in cache_fuse_metadata["check_layers"]:
                    if status == 1:
                        status = 3
                    else:
                        status = 4
            if print_layer_time:
                start = time.time()
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,

                status=status,
                cache_fuse_metadata=cache_fuse_metadata,
                old_kv=old_kv,
            )
            if print_layer_time:
                print(f'attention_time: {time.time()-start}')
                start = time.time()
                    
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            # mlp()
            if print_layer_time:
                print(f'mlp_time: {time.time()-start}')
        
        

        if kv_schedule is None or not loading_computation_parallel:
            return hidden_states, residual
        else:
            return hidden_states, residual, old_kv_next_layer
    
    def calculate_recomputation_modules(self,layer_idx,attention_x,cache_fuse_metadata):
        # output.view(-1, self.num_heads * self.head_size)
        # (1,head,seq_len,seq_len)
        ## =============================================
        ## THE SHAPE OF ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!
        ## =============================================
        # import pdb; pdb.set_trace()
        attention_shape = attention_x.shape
        # ALFRED
        multiple_recompute_panduan = True  # False 14
        compute_attention_itself = False ###False
        with_sink = False # True 18 ###False  18
        diedai = True  ## True 15 ## False 16 
        ref_token_num_neibu = 3
        ref_token_num = 3
        import time
        startx = time.time()
        recompute_num = cache_fuse_metadata["recompute_num"][cache_fuse_metadata["check_layers"].index(layer_idx)]
        recompute_modules = cache_fuse_metadata["recompute_modules"]
        module_lengths = cache_fuse_metadata["module_lengths"]
        last_len = cache_fuse_metadata['suffix_len']
        hidden_state_place = []
        start = 0
        if diedai:
            recompute_slices=[]
            attention = torch.zeros((attention_shape[3],attention_shape[3]),device=attention_x.device,dtype=attention_x.dtype)
            attention_x = attention_x.squeeze(0).sum(dim=0)
            for i in range(len(module_lengths)):
                if recompute_modules[i]:
                    hidden_state_place.append([start,start+module_lengths[i][1]-module_lengths[i][0]])
                    start += module_lengths[i][1]-module_lengths[i][0]
                    # attention[:,:,module_lengths[i][0]:module_lengths[i][1]] = attention_x[:,:,hidden_state_place[-1][0]:hidden_state_place[-1][1]]
                    recompute_slices.append([module_lengths[i][0], module_lengths[i][1]])
                else:
                    hidden_state_place.append(None)
                    # attention[:,:,module_lengths[i][0]:module_lengths[i][1]] = torch.zeros((attention_shape[0],attention_shape[1],module_lengths[i][1]-module_lengths[i][0],attention_shape[3]),dtype=attention_x.dtype).to(attention_x.device)
            recompute_slices.append([recompute_slices[-1][-1], recompute_slices[-1][-1]+last_len])
            attention[torch.cat([torch.arange(s[0], s[1]) for s in recompute_slices]), :] = attention_x
            print(f'prepare_time: {time.time()-startx}')
        else:
            # attention = torch.zeros((attention_shape[0],attention_shape[1],attention_shape[3],attention_shape[3]),device=attention_x.device,dtype=attention_x.dtype)
            for i in range(len(module_lengths)):
                if recompute_modules[i]:
                    hidden_state_place.append([start,start+module_lengths[i][1]-module_lengths[i][0]])
                    start += module_lengths[i][1]-module_lengths[i][0]
                    # attention[:,:,module_lengths[i][0]:module_lengths[i][1]] = attention_x[:,:,hidden_state_place[-1][0]:hidden_state_place[-1][1]]
                else:
                    hidden_state_place.append(None)
                    # attention[:,:,module_lengths[i][0]:module_lengths[i][1]] = torch.zeros((attention_shape[0],attention_shape[1],module_lengths[i][1]-module_lengths[i][0],attention_shape[3]),dtype=attention_x.dtype).to(attention_x.device)
                # attention[:,:,-last_len:, :] = attention_x[:,:,-last_len:,:]
            print(f'prepare_time: {time.time()-startx}')
        if not diedai:
            if multiple_recompute_panduan:
                if compute_attention_itself:
                    attention_sum_lists = [attention_x[:,:,place[1]-ref_token_num_neibu:place[1]]*(module_lengths[-1][-1]-hidden_state_place[-1][-1]+place[1])/module_lengths[-1][-1] for place in hidden_state_place if place is not None]
                else:
                    try:
                        attention_sum_lists = [F.pad(attention_x[:,:,(place[1]-ref_token_num_neibu):place[1],:module_lengths[j][0]], (0, attention_shape[3]-module_lengths[j][0], 0, 0, 0, 0, 0, 0))*(module_lengths[-1][-1]-hidden_state_place[-1][-1]+place[1])/module_lengths[-1][-1] for j,place in enumerate(hidden_state_place) if place is not None]
                        # attention_sum_lists = [F.pad(attention[:,:,(place[1]-ref_token_num_neibu):place[1],:place[0]], (0, attention.shape[3]-place[0], 0, 0, 0, 0, 0, 0))*(total_length-hidden_states.shape[1]+place[1])/total_length for place in hidden_state_place if place is not None]
                    except:
                        import pdb; pdb.set_trace()
                attention_sum_lists.append(attention_x[:,:,-1*ref_token_num:])
                attention_sum_list = torch.cat(attention_sum_lists,dim=2)
                attention_sum = attention_sum_list.sum(dim=1).sum(dim=1)/attention_shape[1]/ref_token_num  # (batch_size, seq_len)
            else:
                attention_sum_list = attention_x[:,:,-1*ref_token_num:]
                attention_sum = attention_sum_list.sum(dim=1).sum(dim=1)/attention_shape[1]/ref_token_num  # (batch_size, seq_len)
            if recompute_modules.any():
                if multiple_recompute_panduan:
                    if with_sink:
                        sink_length = 0
                    else:
                        sink_length = 1
                    if compute_attention_itself:
                        module_attention = torch.tensor([attention_sum[:, length[0]+sink_length:length[1]].sum(dim=1)[0]/(len(module_lengths)-j+1) for j,length in enumerate(module_lengths)]).to('cpu')  # (num_modules)
                    else:
                        module_attention = torch.tensor([attention_sum[:, length[0]+sink_length:length[1]].sum(dim=1)[0]/(len(module_lengths)-j) for j,length in enumerate(module_lengths)]).to('cpu')  # (num_modules)
                else:
                    module_attention = torch.tensor([attention_sum[:, length[0]:length[1]].sum(dim=1)[0] for j,length in enumerate(module_lengths)]).to('cpu')  # (num_modules)
                
                module_attention = torch.where(recompute_modules==1,module_attention,-100)
                module_attention_sorted = torch.sort(module_attention, descending=True)[0]
                recompute_threshould = max(0,module_attention_sorted[max(1, int(recompute_num))-1])
                recompute_modules = torch.where((module_attention>=recompute_threshould) & (recompute_modules==1), torch.ones_like(module_attention, dtype=torch.bool), torch.zeros_like(module_attention, dtype=torch.bool))

# time_point2: 0.0002052783966064453                                                                                                                                                                             | 0/1 [00:00<?, ?it/s]
# time_point2_1: 0.0008220672607421875
# time_point3: 0.0009477138519287109
# time_point4: 0.0013072490692138672
# time_point2: 0.002507925033569336
# time_point2_1: 0.0028896331787109375
# time_point3: 0.002969980239868164
# time_point4: 0.0031528472900390625
# time_point2: 0.004242658615112305
# time_point2_1: 0.004609823226928711
# time_point3: 0.004685401916503906
# time_point4: 0.004863262176513672
# coput_logits_time: 0.0018444061279296875

        if diedai and recompute_modules.any():
            if with_sink:
                sink_length = 0
            else:
                sink_length = 1
            
            end3 = time.time()
            recompute_modules_tmp_old = torch.ones_like(recompute_modules)
            recompute_modules_tmp = torch.zeros_like(recompute_modules)
            module_attention = None
            while not torch.all(recompute_modules_tmp_old == recompute_modules_tmp):
                attention_sum_lists = []
                if module_attention is not None:
                    attention_sum_lists = [torch.cat([F.pad(attention[(place[1]-ref_token_num_neibu):place[1],:module_lengths[j][0]], (0, place[1]-place[0], 0, 0)),attention[place[1]:,(module_lengths[j][1]-ref_token_num_neibu):module_lengths[j][1]].transpose(0,1)],dim=1)*module_attention[j]*10 for j,place in enumerate(module_lengths) if hidden_state_place[j] is not None]
                attention_sum_lists.append(attention[-1*ref_token_num:])
                if len(attention_sum_lists)>1:
                    attention_sum_list = torch.cat(attention_sum_lists,dim=0)
                else:
                    attention_sum_list = attention_sum_lists[0]
                attention_sum = attention_sum_list.sum(dim=0)/attention_shape[1]/ref_token_num  # (seq_len)
                print("time_point2:",time.time()-end3)
                module_attention = torch.tensor([attention_sum[length[0]+sink_length:length[1]].sum(dim=0) for j,length in enumerate(module_lengths)]).to('cpu')  # (num_modules)
                print("time_point2_1:",time.time()-end3)
                # import pdb; pdb.set_trace()
                module_attention = torch.where(recompute_modules==1,module_attention,-100)
                module_attention_sorted = torch.sort(module_attention, descending=True)[0]
                print("time_point3:",time.time()-end3)
                recompute_threshould = max(0,module_attention_sorted[max(1, int(recompute_num))-1])
                recompute_modules_tmp_old = copy.deepcopy(recompute_modules_tmp)
                recompute_modules_tmp = torch.where((module_attention>=recompute_threshould) & (recompute_modules==1), torch.ones_like(module_attention, dtype=torch.bool), torch.zeros_like(module_attention, dtype=torch.bool))
                print("time_point4:",time.time()-end3)
            recompute_modules = recompute_modules_tmp
            # import pdb;pdb.set_trace()
        start2 = time.time()
        cache_fuse_metadata["recompute_modules"] = recompute_modules
        cache_fuse_metadata["imp_indices"] = torch.zeros((cache_fuse_metadata['module_lengths'][-1][-1],), dtype=torch.bool)
        print('chushihua:', time.time()-start2)
        start2 = time.time()
        for i in range(len(module_lengths)):
            if recompute_modules[i]:
                cache_fuse_metadata["imp_indices"][module_lengths[i][0]:module_lengths[i][1]] = 1
        # cache_fuse_metadata["query_imp_indices"] = torch.ones((hidden_state_place[-1][-1]+cache_fuse_metadata['suffix_len'],), dtype=torch.bool)
        cache_fuse_metadata["query_imp_indices"] = torch.ones((attention_shape[2],), dtype=torch.bool)
        print('change_indices:', time.time()-start2)
        start2 = time.time()
        for i in range(len(hidden_state_place)):
            if hidden_state_place[i] is not None and not recompute_modules[i]:
                cache_fuse_metadata["query_imp_indices"][hidden_state_place[i][0]:hidden_state_place[i][1]] = 0    
        print('change_query_indices:', time.time()-start2)  
        print(f'all_time: {time.time()-startx}')
        # import pdb; pdb.set_trace()



class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        
        self.old_kvs = [[None,None]] * len(self.layers)
        self.kv_schedule = None
        self.cache_fuse_metadata = {"check_layers": [],
                                    "check": False,
                                    "recomp_ratios":[0.16],
                                    "recomp_ratio":0.16,
                                    "original_slot_mapping":None,
                                    "our_slot_mapping":None,
                                    "kv_cache_dtype": None,
                                    "attn_bias": None,
                                    "imp_indices": None,
                                    "org_seq_len": None,
                                    "collect": False,
                                    "recompute_num": [],
                                    "recompute_modules": [],
                                    "recompute_delay_ratio": 0.1,
                                    "module_lengths": [],
                                    "query_imp_indices": None,
                                    'suffix_len': None,
                                    "finish_load":False}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if print_layer_time:
            import time
            start1 = time.time()
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        self.cache_fuse_metadata['finish_load'] = False
        if attn_metadata.prefill_metadata:
            temp_status = 0 # full prefill
            if self.cache_fuse_metadata["check"]:
                self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0] 
                check_layer_idx = 0
                self.cache_fuse_metadata["fake_q"] = None  
                self.cache_fuse_metadata["attn_bias"] = None
                self.cache_fuse_metadata["imp_indices"] = None
                self.cache_fuse_metadata["original_slot_mapping"] = None
                self.cache_fuse_metadata["our_slot_mapping"] = None
                self.cache_fuse_metadata['org_pos'] = positions[:]
                
            #FIXME( ): fix this clone for faster time (Is this still needed?)
            #self.cache_fuse_metadata["our_slot_mapping"] = input_metadata.slot_mapping.clone()
        else:
            temp_status = -1 # decode
        residual = None
        if self.cache_fuse_metadata["check"]:
            recompute_num = min(self.kv_schedule.kv_capacity,len(self.kv_schedule.messages_objects))+len(self.kv_schedule.messages_examples)
            # import pdb;pdb.set_trace()
            recompute_num_old = recompute_num
            self.cache_fuse_metadata["recompute_modules"] = torch.ones((recompute_num+1,), dtype=torch.bool)
            self.cache_fuse_metadata["recompute_modules"][0] = 0
            self.cache_fuse_metadata["imp_indices"] = torch.ones((self.cache_fuse_metadata['module_lengths'][-1][-1],), dtype=torch.bool)
            self.cache_fuse_metadata["imp_indices"][:self.cache_fuse_metadata['module_lengths'][0][-1]] = 0
            # import pdb;pdb.set_trace()
            if len(self.cache_fuse_metadata["recompute_num"])==0:
                for i in range(1,len(self.layers)):
                    recompute_num *= self.cache_fuse_metadata["recompute_delay_ratio"]
                    if max(1, int(recompute_num))<recompute_num_old:
                        self.cache_fuse_metadata["recompute_num"].append(max(1, int(recompute_num)))
                        recompute_num_old = max(1, int(recompute_num))
                        self.cache_fuse_metadata["check_layers"].append(i)

        if loading_computation_parallel and self.cache_fuse_metadata["check"]:
            for i in range(len(self.layers)):
                # print(i)
                if print_layer_time:
                    import time
                    start=time.time()
                if self.cache_fuse_metadata["check"]:
                    if i in self.cache_fuse_metadata["check_layers"]:
                        temp_status = 1 # check this layer
                        # attention = self.layers[i].self_attn.attention
                        self.cache_fuse_metadata["check_layer"] = self.cache_fuse_metadata["check_layers"][check_layer_idx]
                        check_layer_idx += 1
                    elif i > self.cache_fuse_metadata["check_layers"][0]:
                        temp_status = 2 # after check
                
                if i==0:
                    old_kv = self.kv_schedule.move_kv_layer_wise_vllm_v1(i,self.cache_fuse_metadata["recompute_modules"])
                if temp_status==1:
                        #import pdb
                        #pdb.set_trace()
                        positions = positions[self.cache_fuse_metadata["query_imp_indices"]]
                layer = self.layers[i]
                hidden_states, residual, old_kv = layer(
                    positions,
                    hidden_states,
                    kv_caches[i],
                    attn_metadata,
                    residual,

                    status = temp_status,
                    cache_fuse_metadata=self.cache_fuse_metadata,
                    old_kv=old_kv,
                    layer_id = i,
                    kv_schedule=self.kv_schedule,
                )

                    

                if print_layer_time:
                    print(f'layer {i} time: {time.time()-start}')
            hidden_states, _ = self.norm(hidden_states, residual)
        else:     
            for i in range(len(self.layers)):
                # print(i)
                if print_layer_time:
                    import time
                    start = time.time()
                if self.cache_fuse_metadata["check"]:
                    if print_layer_time:
                        start2 = time.time()
                    if i in self.cache_fuse_metadata["check_layers"]:
                        temp_status = 1 # check this layer
                        attention = self.layers[i-1].self_attn.attn.impl.attention_weights
                        # output.view(-1, self.num_heads * self.head_size)
                        self.layers[i].calculate_recomputation_modules(i,attention,self.cache_fuse_metadata)

                        old_kv = self.kv_schedule.move_kv_layer_wise_vllm_v1(i,self.cache_fuse_metadata["recompute_modules"])
                        self.cache_fuse_metadata["check_layer"] = self.cache_fuse_metadata["check_layers"][check_layer_idx]
                        check_layer_idx += 1
                        
                    # elif i > self.cache_fuse_metadata["check_layers"][0]:
                    else:
                        temp_status = 2 # after check
                        old_kv = self.kv_schedule.move_kv_layer_wise_vllm_v1(i,self.cache_fuse_metadata["recompute_modules"])
                        # import pdb; pdb.set_trace()
                    if print_layer_time:
                        print(f'loading_kv_time: {time.time()-start2}')
                else:
                    old_kv = self.old_kvs[i]

                layer = self.layers[i]
                # try:
                if temp_status==1:
                    #import pdb
                    #pdb.set_trace()
                    positions = positions[self.cache_fuse_metadata["query_imp_indices"]]
                if self.cache_fuse_metadata["check"] and loading_computation_parallel:
                    hidden_states, residual, _ = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                        attn_metadata,
                        residual,

                        status = temp_status,
                        cache_fuse_metadata=self.cache_fuse_metadata,
                        old_kv=old_kv,
                        layer_id = i,
                        kv_schedule=self.kv_schedule,
                    )
                elif self.cache_fuse_metadata["check"]:
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                        attn_metadata,
                        residual,

                        status = temp_status,
                        cache_fuse_metadata=self.cache_fuse_metadata,
                        old_kv=old_kv,
                        layer_id = i,
                        kv_schedule=self.kv_schedule,
                    )
                else:
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        kv_caches[i],
                        attn_metadata,
                        residual,

                        status = temp_status,
                        cache_fuse_metadata=self.cache_fuse_metadata,
                        old_kv=old_kv
                    )
                # except:
                #     import pdb; pdb.set_trace()
                # try:
                

                if print_layer_time:
                    print(f'layer {i} time: {time.time()-start}')
            hidden_states, _ = self.norm(hidden_states, residual)
            self.old_kvs = [[None,None]] * len(self.layers)
        if print_layer_time:
            print(time.time()-start1)
        return hidden_states




class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = Qwen2Model(config, linear_method)

        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size)
            self.lm_head_weight = self.lm_head.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.hidden_states = None
        self.final_input = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        global print_layer_time
        if self.model.cache_fuse_metadata["check"]:
            # print_layer_time = True
            import time
            start = time.time()
            # import pdb;pdb.set_trace()
            self.model.cache_fuse_metadata["suffix_len"] = input_ids.shape[0]
            module_lengths,token_ids, positions_memory,final_input = self.model.kv_schedule.get_input_ids_vllm()
            self.final_input = final_input
            input_ids = torch.cat([token_ids[0], input_ids], dim=0)
            self.model.cache_fuse_metadata["module_lengths"] = module_lengths
            if self.model.kv_schedule.memory_change == None:
                self.model.kv_schedule.reset_start()
            else:
                # import pdb;pdb.set_trace()
                self.model.kv_schedule.reset_step()
            # import pdb; pdb.set_trace()
            positions_memory.append(torch.tensor(list(range(module_lengths[-1][-1],module_lengths[-1][-1]+self.model.cache_fuse_metadata["suffix_len"]))))
            positions = torch.cat(positions_memory, dim=0).to(positions.device)
            print(f'prepare time: {time.time()-start}')
            # import pdb; pdb.set_trace()
        # import pdb;pdb.set_trace()
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        print_layer_time = False
        self.hidden_states = hidden_states
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits
    
    def compute_logits_vllm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # logits = self.logits_processor(self.lm_head_weight, hidden_states)
        # import pdb; pdb.set_trace()
        return hidden_states @ self.lm_head_weight.t()

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
