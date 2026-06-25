import csv
import os
import collections
import sys
import re

try:
    import javalang
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

def get_error_category(msg, code):
    """Categorize javalang error messages and check for common artifacts."""
    msg_str = str(msg).lower()
    
    # Check for markdown artifacts first
    if '```' in code:
        return "Markdown Artifacts (```)"
    
    if 'expected' in msg_str:
        if ';' in msg_str: return "Missing Semicolon"
        if '}' in msg_str: return "Missing Closing Brace"
        if '{' in msg_str: return "Missing Opening Brace"
        if 'identifier' in msg_str: return "Identifier Expected (Broken ID)"
        return "Expected Symbol"
    
    if 'unexpected' in msg_str: return "Unexpected Token"
    if 'illegal' in msg_str: return "Illegal Start of Expression"
    if 'invalid' in msg_str: return "Invalid Token"
    if 'lexical error' in msg_str or 'could not process token' in msg_str:
        return "Lexical Error (Invalid characters)"
        
    return f"Other: {msg_str[:30]}..."

def analyze_full_integrity(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    print(f"Full analysis of {os.path.basename(file_path)}...")
    
    results = {
        'total': 0,
        'valid': 0,
        'errors': collections.Counter(),
        'error_samples': {} 
    }
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            csv.field_size_limit(10**7)
            reader = csv.DictReader(f)
            
            rows = list(reader)
            results['total'] = len(rows)
            
            for i, row in enumerate(rows):
                code = row.get('full_code', "")
                if not code:
                    continue
                
                try:
                    if HAS_JAVALANG:
                        javalang.parse.parse(code)
                    results['valid'] += 1
                except Exception as e:
                    cat = get_error_category(e, code)
                    results['errors'][cat] += 1
                    if cat not in results['error_samples']:
                        results['error_samples'][cat] = str(e)
                
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{results['total']}...")
                    
            return results
            
    except Exception as e:
        print(f"Critical error analyzing {file_path}: {e}")
        return None

def generate_report(data):
    report = []
    for filename, res in data.items():
        if res is None: continue
        
        valid_rate = (res['valid'] / res['total']) if res['total'] > 0 else 0
        report.append(f"## Report for {filename}")
        report.append(f"- **Total Rows**: {res['total']}")
        report.append(f"- **Valid Java Syntax**: {res['valid']} ({valid_rate:.2%})")
        report.append(f"- **Invalid Java Syntax**: {res['total'] - res['valid']} ({1-valid_rate:.2%})")
        report.append("\n### Error Breakdown")
        report.append("| Category | Count | Percentage | Sample javalang Message |")
        report.append("| :--- | :--- | :--- | :--- |")
        
        for cat, count in res['errors'].most_common():
            pct = count / res['total']
            sample = res['error_samples'].get(cat, "").replace('\n', ' ')
            report.append(f"| {cat} | {count} | {pct:.2%} | `{sample[:100]}...` |")
        report.append("\n" + "="*50 + "\n")
    
    return "\n".join(report)

if __name__ == "__main__":
    if not HAS_JAVALANG:
        print("Error: javalang is required.")
        sys.exit(1)

    targets = [
        "results/DreamCoder32_noPrompt_20260206_094435.csv",
        "results/DreamCoder64_noPrompt_20260206_093622.csv",
        "results/DiffuCoder_Base_noPrompt_20260206_091137.csv",
    ]
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_results = {}
    
    for rel_path in targets:
        abs_path = os.path.join(repo_root, rel_path)
        res = analyze_full_integrity(abs_path)
        all_results[os.path.basename(abs_path)] = res
    
    report_content = generate_report(all_results)
    
    report_path = os.path.join(repo_root, "analysis/integrity_full_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Full Java Code Integrity Report\n\n")
        f.write("This report analyzes the structural integrity of Java code refactored by the models.\n\n")
        f.write(report_content)
    
    print(f"\nReport generated at {report_path}")
