import ast
import token

try:
    from asttokens import ASTTokens
except ImportError:
    ASTTokens = None

def extract_python_identifiers(code):
    """
    提取 Python 代码中的函数名、所有参数名（定义 + 函数体内所有的使用处）及其位置。
    会自动过滤保留词和符号。
    """
    results = []
    
    if not ASTTokens:
        # 简单版本的 ast 处理（不支持精确 token 定位）
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results.append({"type": "function_name", "name": node.name, "line": node.lineno, "col": node.col_offset})
                # 收集本函数的参数名
                params = [arg.arg for arg in node.args.args]
                # 遍历函数体找这些参数的引用
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Name) and sub_node.id in params:
                        results.append({"type": "parameter_usage", "name": sub_node.id, "line": sub_node.lineno, "col": sub_node.col_offset})
                    elif isinstance(sub_node, ast.arg):
                        results.append({"type": "parameter_definition", "name": sub_node.arg, "line": sub_node.lineno, "col": sub_node.col_offset})
        return sorted(results, key=lambda x: (x['line'], x['col']))

    # 方案 B: 使用 ASTTokens 获得极致精度
    atok = ASTTokens(code, parse=True)
    tree = atok.tree

    for node in ast.walk(tree):
        # 1. 提取函数定义名称
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name_token = None
            for t in atok.get_tokens(node):
                if t.type == token.NAME and t.string == node.name:
                    name_token = t
                    break
            if name_token:
                results.append({
                    "type": "function_name",
                    "name": node.name,
                    "line": name_token.start[0],
                    "col": name_token.start[1],
                })
            
            # 2. 确定该函数作用域下的所有参数
            # 我们只需要知道哪些名字是参数，以便在函数体内识别它们
            current_params = set()
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    current_params.add(arg.arg)
            
            # 3. 在整个函数体内寻找这些参数的所有引用 (Name 节点)
            # 注意：ast.walk(node) 会包括参数定义本身所在的 node.args
            for sub_node in ast.walk(node):
                # 处理参数定义 (arg 节点)
                if isinstance(sub_node, ast.arg):
                    start, end = atok.get_text_range(sub_node)
                    pos = atok.get_token_from_range(start, end)
                    results.append({
                        "type": "parameter_definition",
                        "name": sub_node.arg,
                        "line": pos.start[0],
                        "col": pos.start[1],
                    })
                # 处理参数使用 (Name 节点)
                elif isinstance(sub_node, ast.Name) and sub_node.id in current_params:
                    # 我们只记录在函数体内的 Name 引用
                    # (由于 ast.walk 的特性，递归处理可能会重复，我们后续通过位置去重)
                    start, end = atok.get_text_range(sub_node)
                    pos = atok.get_token_from_range(start, end)
                    results.append({
                        "type": "parameter_usage",
                        "name": sub_node.id,
                        "line": pos.start[0],
                        "col": pos.start[1],
                    })

    # 去重：基于 (name, line, col, type)
    unique_results = []
    seen = set()
    for item in results:
        key = (item['name'], item['line'], item['col'], item['type'])
        if key not in seen:
            unique_results.append(item)
            seen.add(key)
            
    return sorted(unique_results, key=lambda x: (x['line'], x['col']))

if __name__ == "__main__":
    test_code = """
def calculate_sum(a, b):
    # 下面是参数 a 和 b 的使用
    result = a + b
    print(f"a is {a}, b is {b}")
    return result

async def move_box(x, y):
    x = x + 10
    return x, y
"""
    identifiers = extract_python_identifiers(test_code)
    
    print(f"{'Type':<20} | {'Name':<15} | {'Line':<5} | {'Col':<5}")
    print("-" * 60)
    for item in identifiers:
        print(f"{item['type']:<20} | {item['name']:<15} | {item['line']:<5} | {item['col']:<5}")
