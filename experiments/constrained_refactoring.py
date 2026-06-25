import os
import torch
import sys
import re
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --- Mock torchvision ---
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
# ------------------------

def get_java_identifier_metadata(text, tokenizer, input_ids_tensor):
    """
    1. 使用 tree-sitter 识别 Java 标识符
    2. 将 Token 归类到对应的标识符下，解决碎片化问题
    """
    input_ids = input_ids_tensor[0].tolist()
    
    # 获取标识符字符偏移范围
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        id_ranges = []
        def traverse(node):
            if node.type == 'identifier':
                id_ranges.append((node.start_byte, node.end_byte, text[node.start_byte:node.end_byte]))
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except Exception as e:
        print(f"Tree-sitter 降级中... ({e})")
        
        java_keywords = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
            "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
            "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
            "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
            "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
            "volatile", "while", "true", "false", "null"
        }
        id_ranges = []
        for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text):
            if m.group(0) not in java_keywords:
                id_ranges.append((m.start(), m.end(), m.group(0)))

    # 手动计算 Token 偏移
    token_offsets = []
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
        full = tokenizer.decode(input_ids[:i+1], skip_special_tokens=False)
        token_offsets.append((len(prefix), len(full)))

    # 建立 Token 索引到标识符的映射组
    identifier_groups = []
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    
    for start_byte, end_byte, id_name in id_ranges:
        group_indices = []
        for i, (t_start, t_end) in enumerate(token_offsets):
            # 如果 Token 的中点落在标识符范围内
            t_mid = (t_start + t_end) / 2
            if start_byte <= t_mid < end_byte:
                group_indices.append(i)
                mask[i] = True
        
        if group_indices:
            identifier_groups.append({
                'name': id_name,
                'indices': group_indices
            })
            
    return mask, identifier_groups

def run_constrained_experiment():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"正在加载模型: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')

    code_snippet = """public int process_data(int v1, int v2) {
    int v3 = v1 + v2;
    if (v3 > 100) {
        return v3 * 2;
    }
    return v3;
}"""
    
    print("\n原始代码 (Java):")
    print(code_snippet)
    
    inputs = tokenizer(code_snippet, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    # 获取掩码和分组信息
    identifier_mask, id_groups = get_java_identifier_metadata(code_snippet, tokenizer, input_ids)
    identifier_mask = identifier_mask.to("cuda")
    
    print(f"\n[语义解析成功] 识别出 {len(id_groups)} 个独立标识符，共覆盖 {identifier_mask.sum().item()} 个 Token。")
    
    constrained_input_ids = input_ids.clone()
    constrained_input_ids[0, identifier_mask] = mask_token_id
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"
    out_dir = f"results/constrained_{experiment_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n运行受限扩散... 结果将保存至: {out_dir}")
    with torch.no_grad():
        output = model.diffusion_generate(
            constrained_input_ids,
            attention_mask=inputs.attention_mask.to("cuda"),
            max_length=input_ids.shape[1] + 1,
            steps=256,
            output_history=True,
            return_dict_in_generate=True,
            temperature=0.3
        )
        
    history = output.history
    
    # 保存每一步
    for step_idx, step_tensor in enumerate(history):
        decoded_text = tokenizer.decode(step_tensor[0], skip_special_tokens=True)
        filename = f"data_test_step{step_idx}_{experiment_id}.java"
        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write(decoded_text)

    # 稳定性分析报告（按标识符分组）
    print("\n" + "="*50)
    print("--- 标识符级稳定性分析报告 (Identifier-Level) ---")
    print("="*50)
    
    for group in id_groups:
        orig_name = group['name']
        indices = group['indices']
        
        # 寻找该组何时完全稳定（所有 token 都被填充）
        max_fill_step = -1
        final_group_tokens = []
        
        # 我们追踪第一次所有 token 都不是 MASK 的时刻
        for step_idx, h in enumerate(history):
            current_tokens = [h[0, i].item() for i in indices]
            if mask_token_id not in current_tokens:
                max_fill_step = step_idx
                final_group_tokens = current_tokens
                break
        
        if max_fill_step != -1:
            refactored_name = tokenizer.decode(final_group_tokens).strip()
            refactored_name = re.sub(r'[^A-Za-z0-9_]', '', refactored_name) 
            # 状态判定
            if refactored_name == orig_name:
                status = "保持不变 (STABLE)"
            else:
                status = f"已重构 (REFACTORED) -> '{refactored_name}'"
            
            print(f"标识符: {orig_name:15s} | 稳定步数: Step {max_fill_step:3d} | 最终结果: {status}")
        else:
            print(f"标识符: {orig_name:15s} | 状态: 未能在 256 步内完成填充。")

if __name__ == "__main__":
    run_constrained_experiment()