import os
import torch
import sys
import re
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --- 环境适配：Mock torchvision ---
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

def get_java_identifier_metadata(text, tokenizer, input_ids_tensor):
    input_ids = input_ids_tensor[0].tolist()
    
    # 关键字屏蔽（防止 for, if 等被作为标识符）
    java_keywords = {
        "public", "static", "int", "if", "return", "void", "class", "for", "new", "boolean",
        "private", "protected", "final", "else", "while", "this", "null", "true", "false"
    }
    
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        id_ranges = []
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    id_ranges.append((node.start_byte, node.end_byte, name))
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except Exception as e:
        id_ranges = []
        for m in re.finditer(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text):
            if m.group(0) not in java_keywords:
                id_ranges.append((m.start(), m.end(), m.group(0)))

    # 计算 Token 偏移
    token_offsets = []
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[:i], skip_special_tokens=False)
        full = tokenizer.decode(input_ids[:i+1], skip_special_tokens=False)
        token_offsets.append((len(prefix), len(full)))

    identifier_groups = []
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    
    for start_byte, end_byte, id_name in id_ranges:
        group_indices = []
        for i, (t_start, t_end) in enumerate(token_offsets):
            t_mid = (t_start + t_end) / 2
            if start_byte <= t_mid < end_byte:
                group_indices.append(i)
                mask[i] = True
        if group_indices:
            identifier_groups.append({'name': id_name, 'indices': group_indices})
            
    return mask, identifier_groups

def run_experiment_case(tokenizer, model, label, code_snippet, mask_token_id, experiment_timestamp):
    print(f"\n>>> 正在运行实验案例: [{label}]")
    inputs = tokenizer(code_snippet, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    
    identifier_mask, id_groups = get_java_identifier_metadata(code_snippet, tokenizer, input_ids)
    identifier_mask = identifier_mask.to("cuda")
    
    constrained_input_ids = input_ids.clone()
    constrained_input_ids[0, identifier_mask] = mask_token_id
    
    out_dir = f"results/constrained_{label}_{experiment_timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
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
    # 保存结果
    final_text = tokenizer.decode(history[-1][0], skip_special_tokens=True)
    with open(os.path.join(out_dir, "final_result.java"), "w") as f:
        f.write(final_text)

    # 分析标识符：按唯一名称进行聚合统计
    stats = {} # key: identifier_name, value: list of fill_steps
    final_names = {} # key: identifier_name, value: list of results
    
    for group in id_groups:
        orig = group['name']
        indices = group['indices']
        fill_step = -1
        final_toks = []
        for s_idx, h in enumerate(history):
            current = [h[0, i].item() for i in indices]
            if mask_token_id not in current:
                fill_step = s_idx
                final_toks = current
                break
        
        if fill_step != -1:
            raw_res = tokenizer.decode(final_toks).strip()
            cleaned_res = re.sub(r'[^A-Za-z0-9_]', '', raw_res)
            
            if orig not in stats:
                stats[orig] = []
                final_names[orig] = []
            
            stats[orig].append(fill_step)
            final_names[orig].append(cleaned_res)

    # 聚合汇总
    aggregated_results = []
    for name in stats:
        avg_step = sum(stats[name]) / len(stats[name])
        # 取出现次数最多的重构名称作为代表
        most_common_name = max(set(final_names[name]), key=final_names[name].count)
        status = "Stable" if most_common_name == name else "Refactored"
        
        aggregated_results.append({
            'original': name,
            'avg_step': f"{avg_step:.1f}",
            'count': len(stats[name]),
            'status': status,
            'result': most_common_name
        })
    return aggregated_results

def run_comparison():
    model_id = "apple/DiffuCoder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 对照组案例
    test_cases = [
        {
            "label": "Bad_Naming",
            "code": """public class Util {
    public static int[] doIt(int[] arr) {
        int l = arr.length;
        int t = 0; 
        for (int i = 0; i < l; i++) {
            for (int j = 1; j < (l - i); j++) {
                if (arr[j - 1] > arr[j]) {
                    t = arr[j - 1];
                    arr[j - 1] = arr[j];
                    arr[j] = t;
                }
            }
        }
        return arr;
    }
}"""
        },
        {
            "label": "Good_Naming",
            "code": """public class Sorter {
    public static int[] bubbleSort(int[] numbers) {
        int n = numbers.length;
        int temp = 0; 
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < (n - i); j++) {
                if (numbers[j - 1] > numbers[j]) {
                    temp = numbers[j - 1];
                    numbers[j - 1] = numbers[j];
                    numbers[j] = temp;
                }
            }
        }
        return numbers;
    }
}"""
        }
    ]

    all_results = {}
    for case in test_cases:
        res = run_experiment_case(tokenizer, model, case['label'], case['code'], mask_token_id, ts)
        all_results[case['label']] = res

    # 打印对比报告
    print("\n" + "="*95)
    print(f"{'CASE':15} | {'IDENTIFIER':15} | {'AVG STEP':10} | {'COUNT':6} | {'STATUS':10} | {'FINAL NAME'}")
    print("-" * 95)
    for label, res_list in all_results.items():
        for r in res_list:
            print(f"{label:15} | {r['original']:15} | {r['avg_step']:10} | {r['count']:6} | {r['status']:10} | {r['result']}")
    print("="*95)

if __name__ == "__main__":
    run_comparison()