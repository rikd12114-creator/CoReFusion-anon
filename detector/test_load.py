import pandas as pd
import ast
from pathlib import Path

def load_developer_dataset(csv_path: Path, max_samples: int = None):
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    samples = []
    for idx, row in df.iterrows():
        tv_str = str(row.get("Target-Var", ""))
        if not tv_str.strip():
            continue
        try:
            tv = ast.literal_eval(tv_str)
            name = list(tv.keys())[0]
            indices = tv[name]
        except Exception:
            continue
            
        code = str(row.get("methodBody", ""))
        
        # We need to replace backward to not mess up offsets
        # Some indices might be slightly off if Test-Dev.csv offsets are bytes or after some normalization
        # Let's verify if the name matches at the offset
        for i in sorted(indices, reverse=True):
            if code[i:i+len(name)] == name:
                code = code[:i] + "[MASK]" + code[i+len(name):]
            else:
                # If offset is wrong, do a simple replace matching whole word?
                import re
                code = re.sub(rf'\b{re.escape(name)}\b', "[MASK]", code)
                break
                
        samples.append({
            "id": idx,
            "name": name,
            "code_snippet": code[:100].replace('\n', ' ')
        })
    print(f"Loaded {len(samples)} samples")
    print("Sample 0:", samples[0])

load_developer_dataset(Path("/path/to/CoReFusion/data/Developer/Test-Dev.csv"), 5)
