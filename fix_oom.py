import sys
import re

for filename in ['experiments/experiment_a_internal_rep.py', 'experiments/experiment_b_unmasking_order.py', 'experiments/experiment_c_token_ranking.py']:
    print(f"Modifying {filename}...")
    with open(filename, "r") as f:
        lines = f.readlines()
        
    out_lines = []
    in_loop = False
    loop_indent = ""
    for i, line in enumerate(lines):
        # Optimize memory usage in add_gumbel_noise
        if "logits = logits.to(torch.float64)" in line:
            line = line.replace("float64", "float32")
        if "dtype=torch.float64" in line:
            line = line.replace("float64", "float32")
            
        if re.match(r"^(\s*)for idx, row in tqdm\(df\.iterrows", line):
            in_loop = True
            loop_indent = re.match(r"^(\s*)", line).group(1)
            out_lines.append(line)
            out_lines.append(loop_indent + "    try:\n")
            continue
            
        if in_loop:
            indentation = len(line) - len(line.lstrip())
            # Find the end of the main loop
            if line.strip() != "" and indentation <= len(loop_indent) and not line.startswith(loop_indent + " "):
                in_loop = False
                out_lines.append(loop_indent + "    except Exception as e:\n")
                out_lines.append(loop_indent + "        if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():\n")
                out_lines.append(loop_indent + "            print(f'\\n[OOM] Skipping sample {row.get(\"id\", \"unknown\")}...')\n")
                out_lines.append(loop_indent + "            torch.cuda.empty_cache()\n")
                out_lines.append(loop_indent + "            import gc; gc.collect()\n")
                out_lines.append(loop_indent + "            continue\n")
                out_lines.append(loop_indent + "        else:\n")
                out_lines.append(loop_indent + "            raise e\n\n")
                out_lines.append(line)
            else:
                if line.strip() == "":
                    out_lines.append("\n")
                else:
                    out_lines.append("    " + line)
        else:
            out_lines.append(line)
            
    with open(filename, "w") as f:
        f.writelines(out_lines)
