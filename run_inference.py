import pandas as pd
import os
import torch
import gc
import sys
import time
from datetime import datetime
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a simple pass-through
    def tqdm(iterable, **kwargs):
        return iterable

# 导入框架中的注册表
from unified_framework import MODEL_REGISTRY

def main():
    # 1. 配置路径
    # 获取脚本所在的根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(root_dir, 'data', 'test.csv')
    output_dir = os.path.join(root_dir, 'results')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. 读取数据
    print(f"正在读取数据: {input_file} ...")
    try:
        # 移除了 nrows 限制以处理所有数据
        df = pd.read_csv(input_file, header=None, names=['id', 'X', 'y'])
        print(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 3. 待测试的模型列表
    # 用户要求对比：DiffuCoder, LLADA, DreamCoder 以及自回归模型 DeepSeek, CodeGemma, Qwen
    models_to_test = ['diffucoder', 'llada', 'dreamcoder', 'deepseek', 'codegemma', 'qwen']
    
    for model_key in models_to_test:
        if model_key not in MODEL_REGISTRY:
            print(f"\n跳过 {model_key}: 不在 MODEL_REGISTRY 中")
            continue
            
        print(f"\n{'='*20} 正在加载模型: {model_key} {'='*20}")
        
        config = MODEL_REGISTRY[model_key]
        try:
            model_instance = config["class"](config["id"])
            model_instance.load()
        except Exception as e:
            print(f"加载模型 {model_key} 失败: {e}")
            continue

        # 获取底层模型和分词器
        raw_model = model_instance.model
        tokenizer = model_instance.tokenizer

        # 4. 执行推理 (使用 Batch 加速)
        BATCH_SIZE = 32 # 针对 A100 80GB 可尝试 32 或更高
        print(f"开始为 {model_key} 执行去噪/生成测试 (Batch Size: {BATCH_SIZE})...")

        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if hasattr(raw_model, "resize_token_embeddings"):
                    raw_model.resize_token_embeddings(len(tokenizer)) 

        # 设置 padding side
        tokenizer.padding_side = 'left'

        total_rows = len(df)
        import math
        num_batches = math.ceil(total_rows / BATCH_SIZE)
        results = []

        # 使用 tqdm 显示进度条
        for i in tqdm(range(0, total_rows, BATCH_SIZE), desc=f"[{model_key}] Inference", unit="batch"):
            chunk = df.iloc[i : i+BATCH_SIZE]
            batch_x = chunk['X'].astype(str).tolist()
            batch_y = chunk['y'].astype(str).tolist()
            batch_ids = chunk['id'].tolist()
            current_batch_idx = (i // BATCH_SIZE) + 1
            
            print(f"[{model_key}] 正在处理批次 {current_batch_idx}/{num_batches} (Ids: {batch_ids[0]} - {batch_ids[-1]})...")
            
            start_time = time.time()
            batch_outputs = []
            
            try:
                if model_key == 'llada':
                    mask_id = 126336
                    from unified_framework import llada_generate
                    if llada_generate is None:
                        raise ImportError("无法加载 llada 的 generate 模块")
                    llada_generate_func = llada_generate

                    input_texts = [x.replace('[MASK]', tokenizer.decode([mask_id])) for x in batch_x]
                    
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    input_ids = inputs.input_ids.to(raw_model.device)
                    attention_mask = inputs.attention_mask.to(raw_model.device)
                    
                    m_ids = tokenizer.encode(tokenizer.decode([mask_id]), add_special_tokens=False)
                    for m_id in m_ids:
                        input_ids[input_ids == m_id] = mask_id

                    output = llada_generate_func(
                        raw_model, 
                        input_ids, 
                        attention_mask=attention_mask,
                        steps=128, 
                        gen_length=1, 
                        block_length=1,
                        mask_id=mask_id
                    )
                    
                    processed_ids = output[:, :input_ids.shape[1]]
                    batch_outputs = tokenizer.batch_decode(processed_ids, skip_special_tokens=True)

                elif model_key in ['deepseek', 'codegemma', 'qwen']:
                    prompts = [f"Please give me the [MASK] token in the following text:\n{x}" for x in batch_x]
                    
                    if model_key == 'codegemma' and hasattr(model_instance.model, 'chat'):
                        # CodeGemma (vLLM)
                        from vllm import SamplingParams
                        sampling_params = SamplingParams(max_tokens=128)
                        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
                        outputs = model_instance.model.chat(messages_batch, sampling_params=sampling_params)
                        batch_outputs = [o.outputs[0].text for o in outputs]
                    else:
                        # DeepSeek, Qwen (HF)
                        if model_key == 'qwen':
                             msgs_batch = [[
                                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                {"role": "user", "content": p}
                            ] for p in prompts]
                             text_batch = tokenizer.apply_chat_template(msgs_batch, tokenize=False, add_generation_prompt=True)
                             inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(raw_model.device)
                        else:
                             # DeepSeek
                             inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(raw_model.device)
                        
                        generated_ids = raw_model.generate(**inputs, max_new_tokens=128)
                        # Remove prompt
                        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
                        batch_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                else:
                    # DiffuCoder / DreamCoder
                    mask_token = '<|mask|>'
                    input_texts = [x.replace('[MASK]', mask_token) for x in batch_x]
                    
                    actual_mask_id = tokenizer.convert_tokens_to_ids(mask_token)
                    
                    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    input_ids = inputs.input_ids.to(raw_model.device)
                    attention_mask = inputs.attention_mask.to(raw_model.device)

                    if i == 0 and actual_mask_id in input_ids:
                         print(f"  [Batch Check] 成功识别到噪声 Token {mask_token} (ID: {actual_mask_id})")

                    gen_steps = 768 if model_key == 'dreamcoder' else 256
                    gen_temp = 0.1 if model_key == 'dreamcoder' else 0.3
                    
                    output = raw_model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1, 
                        steps=gen_steps,
                        temperature=gen_temp,
                        top_p=0.95,
                        alg="entropy",
                        alg_temp=0.,
                    )
                    
                    seqs = output.sequences if hasattr(output, "sequences") else output
                    batch_outputs = tokenizer.batch_decode(seqs, skip_special_tokens=True)

                batch_time = time.time() - start_time
                avg_time = batch_time / len(batch_x)
                
                for idx, out_text in enumerate(batch_outputs):
                    results.append({
                        'id': batch_ids[idx],
                        'X_original': batch_x[idx],
                        'output_full': out_text,
                        'y_ground_truth': batch_y[idx],
                        'processing_time_sec': round(avg_time, 4)
                    })

            except Exception as e:
                import traceback
                print(f"  Batch 处理出错: {e}")
                # traceback.print_exc()
                batch_time = time.time() - start_time
                for idx in range(len(batch_x)):
                     results.append({
                        'id': batch_ids[idx],
                        'X_original': batch_x[idx],
                        'output_full': f"ERROR: {str(e)}",
                        'y_ground_truth': batch_y[idx],
                        'processing_time_sec': round(batch_time / len(batch_x), 4)
                    })
                
        # 5. 保存结果
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_file = os.path.join(output_dir, f'{model_key}_{timestamp}.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"\n{model_key} 推理完成！结果已保存至: {output_file}")

        # --- 新增: 自动上传到 Hugging Face ---
        if HfApi is not None:
            try:
                api = HfApi()
                repo_id = "anonymous/IdentifierRefactoringRes"
                print(f"正在上传 {output_file} 到 Hugging Face 仓库: {repo_id} ...")
                api.upload_file(
                    path_or_fileobj=output_file,
                    path_in_repo=f"results/{os.path.basename(output_file)}",
                    repo_id=repo_id,
                    repo_type="dataset", 
                )
                print(f"上传成功！")
            except Exception as upload_err:
                print(f"上传失败: {upload_err}")
                print("请确保已运行 `huggingface-cli login` 或设置了 HF_TOKEN 环境变量")
        else:
            print("未检测到 huggingface_hub 库，跳过自动上传。")
        # -----------------------------------

        # 6. 清理内存，防止 OOM
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n所有指定模型测试完成！")

if __name__ == "__main__":
    main()