
import os
import sys
import subprocess

try:
    import javalang
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

def check_syntax_javalang(file_path):
    """使用 javalang 解析器检查语法错误"""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        javalang.parse.parse(code)
        return True, "OK"
    except javalang.parser.JavaSyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_syntax_javac(file_path):
    """使用系统 javac 命令进行语法检查 (-proc:none 只检查语法不编译)"""
    try:
        # -proc:none 禁用注解处理, -Xlint:none 关闭警告
        result = subprocess.run(
            ['javac', '-proc:none', '-Xlint:none', file_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, "OK"
        else:
            # 过滤掉 '找不到符号' (error: cannot find symbol) 
            # 因为生成的片段往往缺少 import 或 context，这不属于语法错误
            errors = [line for line in result.stderr.split('\n') if 'error:' in line]
            syntax_errors = [e for e in errors if 'cannot find symbol' not in e and 'package' not in e]
            if not syntax_errors:
                return True, "OK (Missing dependencies but syntax seems fine)"
            return False, "\n".join(syntax_errors)
    except FileNotFoundError:
        return None, "javac not found"

def main():
    # 1. 查找目录 (优先检查 results，如果没有则检查上级目录的 results)
    target_dir = 'results'
    if not os.path.exists(target_dir):
        target_dir = os.path.join('..', 'results')
    
    if not os.path.exists(target_dir):
        print(f"错误: 找不到目录 {target_dir}")
        return

    java_files = [f for f in os.listdir(target_dir) if f.endswith('.java')]
    
    if not java_files:
        print(f"在 {target_dir} 中没有找到 .java 文件。")
        return

    print(f"正在检查目录: {os.path.abspath(target_dir)}")
    print(f"检测到 {len(java_files)} 个文件...\n")
    print(f"{'文件名':<50} | {'结果'}")
    print("-" * 70)

    pass_count = 0
    fail_count = 0

    for filename in java_files:
        file_path = os.path.join(target_dir, filename)
        
        # 优先使用 javalang
        if HAS_JAVALANG:
            is_valid, msg = check_syntax_javalang(file_path)
        else:
            # 备选使用 javac
            res = check_syntax_javac(file_path)
            if res[0] is None:
                print("请安装 javalang (pip install javalang) 或确保 javac 在环境变量中以进行检查。")
                return
            is_valid, msg = res

        if is_valid:
            print(f"{filename:<50} | [✓] {msg}")
            pass_count += 1
        else:
            print(f"{filename:<50} | [✗] {msg}")
            fail_count += 1

    print("\n" + "=" * 30)
    print(f"检查完成!")
    print(f"通过: {pass_count}")
    print(f"失败: {fail_count}")
    print("=" * 30)

if __name__ == "__main__":
    main()
