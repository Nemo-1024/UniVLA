#!/usr/bin/env python3
"""
TFRecord文件完整性检测脚本
用于检测VLA训练数据中损坏的TFRecord文件
"""

import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf
from tqdm import tqdm

def find_tfrecord_files(data_root: str) -> List[str]:
    """
    递归查找所有TFRecord文件
    
    Args:
        data_root: 数据根目录路径
        
    Returns:
        TFRecord文件路径列表
    """
    print(f"🔍 在 {data_root} 中搜索TFRecord文件...")
    
    # 支持多种TFRecord文件扩展名
    patterns = [
        "**/*.tfrecord",
        "**/*.tfrecord-*",
        "**/*.tfrecords",
        "**/*.tfdata-*"
    ]
    
    tfrecord_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_root, pattern), recursive=True)
        tfrecord_files.extend(files)
    
    # 去重并排序
    tfrecord_files = sorted(list(set(tfrecord_files)))
    
    print(f"📁 找到 {len(tfrecord_files)} 个TFRecord文件")
    return tfrecord_files

def check_single_file(record_file: str, verbose: bool = True) -> Tuple[bool, str]:
    """
    检查单个TFRecord文件的完整性
    
    Args:
        record_file: TFRecord文件路径
        verbose: 是否显示详细信息
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if verbose:
            print(f"🔍 检查文件: {record_file}")
        
        # 检查文件是否存在
        if not os.path.exists(record_file):
            return False, "文件不存在"
        
        # 检查文件大小
        file_size = os.path.getsize(record_file)
        if file_size == 0:
            return False, "文件为空"
        
        if verbose:
            print(f"   文件大小: {file_size / (1024*1024):.2f} MB")
        
        # 尝试读取文件中的所有记录
        record_count = 0
        dataset = tf.data.TFRecordDataset(record_file)
        
        for raw_record in dataset:
            record_count += 1
            # 每1000条记录显示一次进度（仅在verbose模式下）
            if verbose and record_count % 1000 == 0:
                print(f"   已读取 {record_count} 条记录...")
        
        if verbose:
            print(f"✅ {record_file} 检查通过 (共 {record_count} 条记录)")
        
        return True, f"文件正常，共{record_count}条记录"
        
    except tf.errors.DataLossError as e:
        error_msg = f"数据损坏错误: {str(e)}"
        if verbose:
            print(f"❌ {record_file} - {error_msg}")
        return False, error_msg
    
    except tf.errors.OutOfRangeError as e:
        error_msg = f"数据范围错误: {str(e)}"
        if verbose:
            print(f"❌ {record_file} - {error_msg}")
        return False, error_msg
    
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        if verbose:
            print(f"❌ {record_file} - {error_msg}")
        return False, error_msg

def check_single_tfrecord_file(file_path: str) -> bool:
    """
    快速检查单个TFRecord文件（简化版本）
    
    Args:
        file_path: TFRecord文件路径
        
    Returns:
        True if file is valid, False otherwise
    """
    is_valid, _ = check_single_file(file_path, verbose=False)
    return is_valid

def check_tfrecord_integrity(path: str, 
                           stop_on_first_error: bool = False,
                           verbose: bool = True) -> dict:
    """
    检查指定路径的TFRecord文件完整性（支持单个文件或目录）
    
    Args:
        path: 文件路径或数据目录路径
        stop_on_first_error: 是否在第一个错误时停止
        verbose: 是否显示详细信息
        
    Returns:
        检查结果字典
    """
    print(f"🚀 开始检查TFRecord文件完整性...")
    print(f"📂 目标路径: {path}")
    print(f"⚙️  遇到错误时停止: {stop_on_first_error}")
    print("-" * 60)
    
    # 判断是文件还是目录
    if os.path.isfile(path):
        print(f"🔍 检测到单个文件: {os.path.basename(path)}")
        tfrecord_files = [path]
    elif os.path.isdir(path):
        print(f"🔍 检测到目录，搜索TFRecord文件...")
        tfrecord_files = find_tfrecord_files(path)
    else:
        print(f"❌ 错误: 路径 {path} 既不是文件也不是目录!")
        return {"total": 0, "valid": 0, "corrupted": 0, "corrupted_files": [], "error": "Invalid path"}
    
    if not tfrecord_files:
        print("⚠️  未找到任何TFRecord文件!")
        return {"total": 0, "valid": 0, "corrupted": 0, "corrupted_files": []}
    
    # 检查每个文件
    valid_files = []
    corrupted_files = []
    
    start_time = time.time()
    
    # 使用进度条
    with tqdm(total=len(tfrecord_files), desc="检查文件") as pbar:
        for i, record_file in enumerate(tfrecord_files):
            pbar.set_description(f"检查文件 {i+1}/{len(tfrecord_files)}")
            
            is_valid, error_msg = check_single_file(record_file, verbose=verbose)
            
            if is_valid:
                valid_files.append(record_file)
            else:
                corrupted_files.append((record_file, error_msg))
                
                if stop_on_first_error:
                    print(f"\n❌ 发现第一个损坏文件，停止检查:")
                    print(f"   文件: {record_file}")
                    print(f"   错误: {error_msg}")
                    break
            
            pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("📊 检查结果摘要")
    print("="*60)
    print(f"总文件数: {len(tfrecord_files)}")
    print(f"正常文件: {len(valid_files)}")
    print(f"损坏文件: {len(corrupted_files)}")
    print(f"检查耗时: {elapsed_time:.2f} 秒")
    
    if corrupted_files:
        print(f"\n❌ 发现 {len(corrupted_files)} 个损坏文件:")
        for file_path, error in corrupted_files:
            print(f"   • {file_path}")
            print(f"     错误: {error}")
    else:
        print(f"\n✅ 所有文件检查通过!")
    
    return {
        "total": len(tfrecord_files),
        "valid": len(valid_files),
        "corrupted": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "valid_files": valid_files,
        "elapsed_time": elapsed_time
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TFRecord文件完整性检测工具")
    parser.add_argument(
        "path", 
        help="TFRecord文件路径或数据目录路径 (例如: /path/to/file.tfrecord 或 /data/datasets/oxe_data)"
    )
    parser.add_argument(
        "--stop-on-error", 
        action="store_true",
        help="遇到第一个错误时停止检查"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="静默模式，只显示摘要结果"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="将结果保存到指定文件"
    )
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.path):
        print(f"❌ 错误: 路径 {args.path} 不存在!")
        sys.exit(1)
    
    # 执行检查
    results = check_tfrecord_integrity(
        path=args.path,
        stop_on_first_error=args.stop_on_error,
        verbose=not args.quiet
    )
    
    # 保存结果到文件
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式
            output_data = {
                "summary": {
                    "total_files": results["total"],
                    "valid_files": results["valid"],
                    "corrupted_files": results["corrupted"],
                    "elapsed_time": results["elapsed_time"]
                },
                "corrupted_files": [
                    {"file": file_path, "error": error} 
                    for file_path, error in results["corrupted_files"]
                ],
                "valid_files": results["valid_files"]
            }
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"📄 结果已保存到: {args.output}")
    
    # 设置退出码
    sys.exit(0 if results["corrupted"] == 0 else 1)

if __name__ == "__main__":
    # 如果直接运行，显示使用帮助
    if len(sys.argv) == 1:
        print("🔧 TFRecord文件完整性检测工具")
        print("=" * 50)
        print("用法示例:")
        print(f"  检查单个文件: python {sys.argv[0]} /path/to/file.tfrecord")
        print(f"  检查整个目录: python {sys.argv[0]} /path/to/dataset/")
        print(f"  静默模式:     python {sys.argv[0]} /path/to/dataset/ --quiet")
        print(f"  遇错停止:     python {sys.argv[0]} /path/to/dataset/ --stop-on-error")
        print(f"  保存结果:     python {sys.argv[0]} /path/to/dataset/ --output results.json")
        print("\n支持的文件格式:")
        print("  • *.tfrecord")
        print("  • *.tfrecord-*")
        print("  • *.tfrecords") 
        print("  • *.tfdata-*")
        
        print("\n常见数据路径检查:")
        # 常见的数据路径
        common_paths = [
            "/data/datasets",
            "/data/home/jlchen/datasets", 
            "~/datasets",
            "./data",
        ]
        
        for path in common_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                print(f"  • {expanded_path} ✅")
            else:
                print(f"  • {expanded_path} ❌")
        
        print(f"\n使用 python {sys.argv[0]} --help 查看完整帮助")
        sys.exit(1)
    
    main()

# ============================================================================
# 便捷使用示例 (可以直接在Python中导入使用)
# ============================================================================

def quick_check_examples():
    """
    一些快速检查的示例代码
    """
    
    # 示例1: 检查单个文件
    single_file = "/path/to/your/file.tfrecord"
    if os.path.exists(single_file):
        is_valid = check_single_tfrecord_file(single_file)
        print(f"文件 {single_file} {'正常' if is_valid else '损坏'}")
    
    # 示例2: 批量检查多个文件
    file_list = [
        "/path/to/file1.tfrecord",
        "/path/to/file2.tfrecord", 
        "/path/to/file3.tfrecord"
    ]
    
    for file_path in file_list:
        if os.path.exists(file_path):
            is_valid = check_single_tfrecord_file(file_path)
            print(f"{'✅' if is_valid else '❌'} {file_path}")
    
    # 示例3: 检查目录并获取详细结果
    dataset_dir = "/path/to/your/dataset"
    if os.path.exists(dataset_dir):
        results = check_tfrecord_integrity(dataset_dir, verbose=False)
        print(f"检查完成: {results['valid']}/{results['total']} 文件正常")

# 如果您想在Python脚本中使用，可以这样导入:
# from check_tfrecord_integrity import check_single_tfrecord_file, check_tfrecord_integrity