
import pandas as pd
import os
from datetime import datetime

def export_java_data():
    # 1. 配置路径
    csv_path = os.path.join('../data', 'test.csv')
    output_dir = 'java_samples'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 2. 读取数据
    print(f"正在从 {csv_path} 读取数据...")
    try:
        # 按照项目中 run_inference.py 的逻辑，列分别为 [id, X, y]
        df = pd.read_csv(csv_path, header=None, names=['id', 'X', 'y'], nrows=10)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"开始导出前 {len(df)} 个数据点...")

    for i, row in df.iterrows():
        # 获取原始 ID、输入文本 X 和 标签 y
        original_id = row['id']
        input_text = str(row['X'])
        ground_truth = str(row['y'])
        
        # 将 [MASK] 替换为真实的 y 得到完整的 Java 代码
        full_code = input_text.replace('[MASK]', ground_truth)
        
        # 文件命名格式: 序号_对应的y_日期_时间.java
        # 注意: y 可能会包含不合法的文件字符，这里做简单处理
        safe_y = "".join(x for x in ground_truth if x.isalnum() or x in "._-")
        filename = f"{i}_{safe_y}_{timestamp}.java"
        filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)
            
        print(f"  已导出: {filename} (ID: {original_id})")

    print(f"\n导出完成！所有文件已保存至: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    export_java_data()
