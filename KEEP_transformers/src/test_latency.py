import torch
import time
import numpy as np
from transformers import Qwen2ForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def test_qwen14b_latency():
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    print("正在加载Qwen-14B模型...")
    model_name = "/data1/pretrained_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"  # 或者使用本地路径
    # model_name = "/data1/pretrained_models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=False
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    model.eval()
    
    # 测试配置
    prompt_lengths = [500, 1000, 1500, 2000, 2500, 3000, 3800]
    warmup_steps = 3
    test_runs = 2
    
    results = []
    
    
    for x in prompt_lengths:
        past_kv_length = 3800 - x
        print(f"\n测试配置: prompt_length={x}, past_kv_length={past_kv_length}")
        
        # 生成测试文本
        test_prompt = "这是一段测试文本。" * x  # 简单生成测试文本
        
        # 编码文本
        inputs = tokenizer([test_prompt], return_tensors="pt")
        input_ids = inputs["input_ids"][:,:x].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # # 预热
        # with torch.no_grad():
        #     for _ in range(warmup_steps):
        #         _ = model.generate(
        #             input_ids,
        #             attention_mask=attention_mask,
        #             max_new_tokens=1,
        #             do_sample=False,
        #             pad_token_id=tokenizer.eos_token_id
        #         )

        # 测试推理延迟
        latencies = []
        print("  测试推理延迟...", end=" ")
        
        with torch.no_grad():
            for i in range(test_runs):
                torch.cuda.synchronize() if device == "cuda" else None
                
                
                # 模拟past KV cache存在的情况
                if past_kv_length > 0:
                    # 创建假的past_key_values来模拟已有缓存
                    batch_size = input_ids.shape[0]
                    fake_past = []
                    
                    # 获取模型的层数
                    with torch.no_grad():
                        # 为每层创建假的past key values
                        for layer_idx in range(model.config.num_hidden_layers):
                            hidden_size = model.config.hidden_size
                            num_heads = model.config.num_key_value_heads
                            head_dim = 128
                            
                            # 创建适当形状的假tensor
                            past_key = torch.randn(
                                batch_size, num_heads, past_kv_length, head_dim,
                                device="cpu", dtype=model.dtype
                            )
                            past_value = torch.randn(
                                batch_size, num_heads, past_kv_length, head_dim,
                                device="cpu", dtype=model.dtype
                            )
                            fake_past.append((past_key, past_value))
                        start_loading_time = time.time()
                        for layer_idx in range(model.config.num_hidden_layers):
                            fake_past[layer_idx] = (
                                fake_past[layer_idx][0].to(device),
                                fake_past[layer_idx][1].to(device)
                            )
                        loading_time = time.time() - start_loading_time
                # 实际推理测试
                start_time = time.time()
                if past_kv_length > 0:
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=tuple(fake_past),
                        max_new_tokens=1,
                        use_cache=True
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        max_new_tokens=1,
                        use_cache=True
                    )
                
                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
                
                if (i + 1) % 2 == 0:
                    print(f"{i+1}", end=" ")
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        results.append({
            'prompt_length': x,
            'past_kv_length': past_kv_length,
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'all_latencies': latencies,
            'loading_time_ms': loading_time if past_kv_length > 0 else 0
        })
        
        print(f"\n  平均延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  加载Past KV时间: {loading_time*1000:.2f} ms" if past_kv_length > 0 else "  无Past KV加载时间")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    print(f"{'Prompt长度':<12} {'Past KV长度':<12} {'平均延迟(ms)':<15} {'标准差(ms)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['prompt_length']:<12} {result['past_kv_length']:<12} "
              f"{result['avg_latency_ms']:<15.2f} {result['std_latency_ms']:<12.2f} {result['loading_time_ms']:<15.2f}")
    
    # 绘制图表
    return results



if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    
    results = test_qwen14b_latency()

