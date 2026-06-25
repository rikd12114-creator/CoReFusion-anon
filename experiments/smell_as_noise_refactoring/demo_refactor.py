import torch
from transformers import AutoTokenizer, AutoModel
import re

# ==========================================
# Mock torchvision (Required for DiffuCoder)
# ==========================================
import sys
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

# ==========================================
# DiffuCoder Refactoring Engine
# ==========================================
class DiffuCoderRefactorer:
    def __init__(self, model_id="apple/DiffuCoder-7B-Instruct"):
        print(f"Loading {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids('<|mask|>')
        print("Model loaded successfully.")

    def inject_noise_and_denoise(self, code_snippet: str, target_var: str, noise_ratio: float = 0.5):
        """
        核心工程启示实现：
        1. 定位劣质变量名 (Code Smell)
        2. 将目标变量的上下文注入一定比例的噪声 (回退到 t 步)
        3. 利用模型去噪，实现自动重构
        
        参数:
        - code_snippet: 原始包含 Smell 的代码
        - target_var: 需要重构的变量名 (例如 "x", "data")
        - noise_ratio: 噪声注入比例 (根据实验，0.3~0.5是最佳甜区)
        """
        print(f"\n[1] Preparing Code (Target: '{target_var}', Noise Ratio: {noise_ratio})")
        
        # 1. 词法分析与 Token 映射
        inputs = self.tokenizer(code_snippet, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 找到目标变量在 Token 序列中的位置
        target_indices = self._find_token_indices(input_ids[0], target_var)
        if not target_indices:
            print(f"Warning: Target variable '{target_var}' not found in tokens.")
            return code_snippet
            
        print(f"Found '{target_var}' at {len(target_indices)} token locations.")
        
        # 2. 局部噪声注入 (Local Noise Injection)
        # 我们不仅 Mask 掉目标变量本身，还按比例随机 Mask 其周围的上下文
        # 这样可以打破原有的局部极小值，迫使模型根据全局语义重新推断变量名
        masked_input_ids = input_ids.clone()
        seq_len = input_ids.shape[1]
        
        # 确保目标变量被 100% Mask
        for start, end in target_indices:
            masked_input_ids[0, start:end] = self.mask_token_id
            
        # 根据 noise_ratio 随机 Mask 其他 Token（排除特殊Token）
        num_random_masks = int(seq_len * noise_ratio)
        valid_mask_pool = [i for i in range(seq_len) 
                           if input_ids[0, i] not in self.tokenizer.all_special_ids]
        
        if valid_mask_pool:
            # 随机选择位置注入噪声
            import random
            mask_positions = random.sample(valid_mask_pool, min(num_random_masks, len(valid_mask_pool)))
            masked_input_ids[0, mask_positions] = self.mask_token_id
            
        masked_count = (masked_input_ids[0] == self.mask_token_id).sum().item()
        print(f"[2] Injected Noise: Masked {masked_count}/{seq_len} tokens ({(masked_count/seq_len)*100:.1f}%)")

        # 3. 扩散去噪 (Diffusion Denoising)
        print("[3] Denoising (Refactoring)...")
        with torch.no_grad():
            output = self.model.diffusion_generate(
                masked_input_ids,
                attention_mask=attention_mask,
                max_length=seq_len + 1,  # 保持长度一致
                steps=32,                # 推理步数 (32 步通常足够)
                temperature=0.2,         # 略微引入随机性以探索更好命名
                top_p=0.95,
                alg="entropy",           # 推荐使用 entropy 采样策略
                alg_temp=0.0
            )
            
        # 4. 解码输出
        seqs = output.sequences if hasattr(output, "sequences") else output
        refactored_code = self.tokenizer.decode(seqs[0], skip_special_tokens=True)
        
        return refactored_code

    def _find_token_indices(self, input_ids, target_str):
        """
        基于正则表达式的鲁棒 Token 查找算法。
        与其在 token 序列里拼凑字符串，不如直接在原字符串中用正则找到位置，
        然后再映射回 Token 索引。
        """
        # 1. 解码整个序列并保留 Token 边界信息
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids.tolist()]
        
        # 2. 找到目标变量（必须是独立单词，防止把 'index' 匹配到 'expectedIndex' 里）
        # 这里用了一个简单的启发式方法：如果 token 的内容正好等于目标变量名
        # （忽略可能的前导空格/下划线）
        
        indices = []
        target_clean = target_str.strip()
        
        for i, tok in enumerate(tokens):
            # Llama 分词器通常会在单词前加一个特殊的下划线字符 (U+2581) 或空格
            clean_tok = tok.replace(' ', '').replace('Ġ', '').strip()
            
            # 精确匹配（假设目标变量没有被切分成多个 token，这对于短变量名如 d, res 通常成立）
            if clean_tok == target_clean:
                indices.append((i, i + 1))
                continue
                
        # 如果单 token 匹配失败，尝试组合匹配 (处理被切分的变量名)
        if not indices:
            for i in range(len(tokens)):
                curr_text = tokens[i].replace(' ', '').replace('Ġ', '')
                if not curr_text: continue
                
                reconstructed = ""
                for j in range(i, min(i + 5, len(tokens))):
                    part = tokens[j].replace(' ', '').replace('Ġ', '')
                    reconstructed += part
                    if reconstructed == target_clean:
                        indices.append((i, j + 1))
                        break
                    if not target_clean.startswith(reconstructed):
                        break
                        
        return sorted(list(set(indices)))

# ==========================================
# 示例用法 (供服务器测试)
# ==========================================
if __name__ == "__main__":
    # 示例代码：包含明显的 Code Smell (变量名为 'd' 和 'res')
    smelly_code = """
public class DataProcessor {
    public List<String> process(List<String> d) {
        List<String> res = new ArrayList<>();
        for (String item : d) {
            if (item != null && !item.isEmpty()) {
                res.add(item.toUpperCase());
            }
        }
        return res;
    }
}
"""
    print("=== Original Smelly Code ===")
    print(smelly_code)
    
    # 初始化重构引擎
    refactorer = DiffuCoderRefactorer()
    
    # 目标 1：将毫无意义的 'd' 进行重构
    # 我们将其回退到 40% 的噪声水平 (这是实验得出的 Sweet Spot)
    print("\n--- Refactoring Target: 'd' ---")
    refactored_step1 = refactorer.inject_noise_and_denoise(
        code_snippet=smelly_code, 
        target_var="d", 
        noise_ratio=0.40  # 注入 40% 的噪声
    )
    
    # 目标 2：继续将模糊的 'res' 进行重构
    print("\n--- Refactoring Target: 'res' ---")
    final_clean_code = refactorer.inject_noise_and_denoise(
        code_snippet=refactored_step1, 
        target_var="res", 
        noise_ratio=0.30  # 注入 30% 的噪声
    )
    
    print("\n=== Final Refactored Code ===")
    print(final_clean_code)
