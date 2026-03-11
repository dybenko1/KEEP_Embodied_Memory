# KEEP: A KV-CACHE-CENTRIC MEMORY SYSTEM FOR EMBODIED PLANNING

## Environment

Ubuntu 14.04+ is required. The scripts were developed and tested on Ubuntu 22.04 and Python 3.8.

You can use WSL-Ubuntu on Windows 10/11.

## Install

1. Clone the whole repo.
    ```bash
    $ git clone {repo_url}
    ```

1. Setup a virtual environment.
    ```bash
    $ conda create -n {env_name} python=3.9
    $ conda activate {env_name}
    ```

1. Install PyTorch (2.2.1) first (see https://pytorch.org/get-started/locally/).
    ```bash
    # exemplary install command for PyTorch 2.2.1 with CUDA 12.1
    $ pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    ```

1. Install python packages in `requirements.txt`.
    ```bash
    $ pip install -r requirements.txt
    ```

1. Install `vllm`.
    ```bash
    $ cd vllm_KEEP
    $ pip install -e .
    ```

## Benchmarking on ALFRED

### Download ALFRED dataset.
```bash
$ cd alfred/data
$ sh download_data.sh json
```

### Download Qwen-2.5 models.
Qwen-2.5-7b: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Qwen-2.5-14b: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

Qwen-2.5-32b: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct


### Benchmarking
```bash
$ bash vllm_latency_test.sh
```

### Headless Server

Please run `startx.py` script before running ALFRED experiment on headless servers. Below script uses 1 for the X_DISPLAY id, but you can use different ids such as 0.

```bash
$ sudo python3 alfred/scripts/startx.py 1
```


## FAQ

* Running out of disk space for Huggingface models
  * You can set the cache folder to be in another disk.
    ```bash
    $ export TRANSFORMERS_CACHE=/mnt/otherdisk/.hf_cache/
    ```

* I have encountered 'cannot find X server with xdpyinfo' in running ALFRED experiments.
  * Please try another x_display number (this should be a string; e.g., '1') in the config file.
    ```bash
    $ python src/evaluate.py --config-name=config_alfred alfred.x_display='1'
    ```

