# KEEP: A KV-CACHE-CENTRIC MEMORY SYSTEM FOR EMBODIED PLANNING

KEEP is a high-performance memory management system designed to enable efficient long-horizon task planning for LLM-powered embodied agents. By introducing a KV-cache-centric memory architecture, KEEP significantly reduces inference latency while maintaining planning accuracy in dynamic environments.

## Key Features 🚀 

- **Static-Dynamic Memory Construction**: Groups memory by update frequency to minimize KV cache invalidation

- **Multi-Hop Memory Recomputation**: Dynamically retrieves and recomputes critical memory interactions through iterative importance propagation

- **Layer-Balanced Pipeline Scheduling**: Eliminates computation bubbles via cross-layer prefetching and balanced loading

## Why KEEP? 📚 
While LLMs show great promise for embodied planning, traditional context-window memory approaches lead to long prompts and high latency. KEEP directly addresses this by:

- Enabling >2× faster planning than baseline methods (e.g., on ALFRED benchmark)

- Maintaining competitive task success rates with significantly lower TTFT

- Providing scalable KV memory management for long-horizon tasks

## Install
`KEEP_transformers` is a simple implementation of KEEP based on Transformers for accuracy evaluation.

`KEEP_vllm` is a implementation on vLLM for latency evaluation.

The two implementations require different environments. Please follow the setup instructions inside each respective file.


## Citation
If you find this work useful for your research, please cite our paper:
```bash
@article{yang2026keep,
  title={KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning},
  author={Yang, Zebin and Xie, Tong and Lu, Baotong and Liu, Shaoshan and Yu, Bo and Li, Meng},
  journal={arXiv preprint arXiv:2602.23592},
  year={2026}
}
```
