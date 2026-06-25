import csv
import os
import subprocess
import tempfile
import shutil

def check_syntax_javac(code_content):
    """
    Checks the syntax of Java code using 'javac'.
    Returns (is_valid, error_msg).
    """
    # Create a temporary directory to avoid file naming conflicts and cleanup easily
    tmp_dir = tempfile.mkdtemp()
    # We need a valid Java filename. Since we don't necessarily know the class name, 
    # 'Test.java' is risky if the code defines a public class with a different name.
    # However, 'javac' usually allows checking syntax even if the filename doesn't match the class name 
    # if we use certain flags or if we just want a syntax check.
    # Actually, if there's a 'public class X', it MUST be in 'X.java'.
    
    # Simple heuristic to find public class name
    class_name = "TempCheck"
    for line in code_content.split('\n'):
        if 'public class ' in line:
            parts = line.split('public class ')
            if len(parts) > 1:
                class_name = parts[1].split('{')[0].split()[0].strip()
                break
    
    file_path = os.path.join(tmp_dir, f"{class_name}.java")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # -proc:none disables annotation processing
        # -Xlint:none disables warnings
        result = subprocess.run(
            ['javac', '-proc:none', '-Xlint:none', file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            shutil.rmtree(tmp_dir)
            return True, "OK"
        else:
            # Filter errors: ignore 'cannot find symbol' and 'package/import' related issues
            # because these are usually isolated snippets.
            stderr_lines = result.stderr.split('\n')
            errors = [line for line in stderr_lines if 'error:' in line]
            
            # Syntax errors are things like 'expected', 'illegal start of expression', etc.
            # We filter out typical "missing environment" errors.
            real_errors = []
            for e in errors:
                if 'cannot find symbol' in e: continue
                if 'package' in e and 'does not exist' in e: continue
                if 'import' in e: continue
                if 'is public, should be declared in a file named' in e: continue # Filename mismatch is fine for syntax check
                real_errors.append(e)
            
            shutil.rmtree(tmp_dir)
            if not real_errors:
                return True, "OK (Syntax fine, missing symbols)"
            return False, "; ".join(real_errors[:3]) # Return first few errors
            
    except Exception as e:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        return False, str(e)

def analyze_csv_integrity(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    print(f"Analyzing {os.path.basename(file_path)}...")
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            # Handle potential large fields in CSV
            csv.field_size_limit(10**7)
            reader = csv.DictReader(f)
            
            if 'full_code' not in reader.fieldnames:
                print(f"Error: {file_path} missing 'full_code' column.")
                return None
            
            total = 0
            valid = 0
            
            # We might not want to check ALL rows if the file is huge (e.g. 19k rows)
            # as javac is slow. Let's limit to top 100 or so for a representative sample,
            # or check every Nth row.
            
            rows = list(reader)
            total_rows = len(rows)
            sample_size = min(total_rows, 100) # Check up to 100 samples per file
            
            import random
            sample_rows = random.sample(rows, sample_size) if total_rows > sample_size else rows
            
            for i, row in enumerate(sample_rows):
                code = row['full_code']
                if not code:
                    continue
                
                is_valid, msg = check_syntax_javac(code)
                if is_valid:
                    valid += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{sample_size}...")
            
            integrity_rate = (valid / sample_size) if sample_size > 0 else 0
            return integrity_rate, sample_size
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    results_to_analyze = [
        "results/DreamCoder32_noPrompt_20260206_094435.csv",
        "results/DreamCoder64_noPrompt_20260206_093622.csv",
        "results/DiffuCoder_Base_noPrompt_20260206_091137.csv",
        "results/DeepSeek-Coder-6.7B-Instruct_refineID_results(1).csv",
        "results/Llama-3.1-8B-Instruct_refineID_results(1).csv",
        "results/Qwen2.5-Coder-7B-Instruct_refineID_results(1).csv",
    ]
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"{'Filename':<60} | {'Integrity (Syntax OK)':<20} | {'Sample Size'}")
    print("-" * 100)
    
    for relative_path in results_to_analyze:
        file_path = os.path.join(repo_root, relative_path)
        result = analyze_csv_integrity(file_path)
        if result:
            rate, size = result
            filename = os.path.basename(file_path)
            print(f"{filename:<60} | {rate:.4%} | {size}")
