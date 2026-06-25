import os
import torch
import random
import sys
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --- Mock torchvision (同前) ---
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

def run_refactoring_test():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')

    # 1. 准备带有“低质量命名”的代码
    sample_code = """def calculate_area(radius):
    # 'v1' 是一个质量很差的变量名
    v1 = 3.14159 * radius * radius
    return v1"""
    
    print("Original code with poor naming (v1):")
    print(sample_code)

    # 2. Tokenize
    inputs = tokenizer(sample_code, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    # 3. 主动注入随机噪声 (例如 30% 的 Token 被替换为 Mask)
    # 这样模型就必须根据上下文来“重构”整段代码，而不仅仅是保留输入
    noisy_input_ids = input_ids.clone()
    seq_len = input_ids.shape[1]
    
    # 随机选择 30% 的位置进行 Mask
    mask_indices = random.sample(range(seq_len), int(seq_len * 0.3))
    for idx in mask_indices:
        noisy_input_ids[0, idx] = mask_token_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/refactor_step_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Diffusion with 30% random noise... saving to {output_dir}")

    # 4. 运行采样并保存每一步
    with torch.no_grad():
        output = model.diffusion_generate(
            noisy_input_ids,
            max_length=seq_len + 1,
            steps=256,
            output_history=True,
            return_dict_in_generate=True,
            temperature=0.3
        )

    history = output.history
    for step_idx, step_tensor in enumerate(history):
        step_tokens = step_tensor[0].tolist()
        decoded_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
        
        # 我们可以重点观察 v1 是否被重写为了 area 或 result
        filename = f"step_{step_idx:03d}.java"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(decoded_text)

    print(f"Done. Please check {output_dir} to see if 'v1' was refactored earlier or later than other tokens.")

if __name__ == "__main__":
    run_refactoring_test()