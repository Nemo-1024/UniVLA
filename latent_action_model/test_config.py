#!/usr/bin/env python3
"""测试配置文件解析"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.lam_lightinng import _get_class_from_string

def test_class_import():
    """测试类导入功能"""
    print("测试简短类名导入...")
    
    # 测试简短类名
    encoder_class = _get_class_from_string("LAMEncoder")
    print(f"LAMEncoder: {encoder_class}")
    
    encoder_v2_class = _get_class_from_string("LAMEncoder_v2")
    print(f"LAMEncoder_v2: {encoder_v2_class}")
    
    decoder_v2_class = _get_class_from_string("LAMDecoder_v2")
    print(f"LAMDecoder_v2: {decoder_v2_class}")
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_class_import()
