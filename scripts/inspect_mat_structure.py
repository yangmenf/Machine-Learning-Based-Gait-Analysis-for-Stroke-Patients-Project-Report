#!/usr/bin/env python3
"""
MAT文件结构检查脚本
详细检查MAT文件中的所有数据组和受试者数量
"""

import os
import sys
import subprocess

def install_compatible_packages():
    """安装兼容的包版本"""
    print("正在安装兼容的包版本...")
    
    packages_to_install = [
        "numpy==1.24.3",
        "h5py==3.8.0", 
        "scipy==1.10.1"
    ]
    
    for package in packages_to_install:
        try:
            print(f"安装 {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ {package} 安装成功")
            else:
                print(f"✗ {package} 安装失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"✗ {package} 安装超时")
        except Exception as e:
            print(f"✗ {package} 安装出错: {e}")

def try_import_libraries():
    """尝试导入必要的库"""
    libraries = {}
    
    try:
        import numpy as np
        libraries['numpy'] = np
        print(f"✓ NumPy {np.__version__} 导入成功")
    except Exception as e:
        print(f"✗ NumPy 导入失败: {e}")
        return None
    
    try:
        import h5py
        libraries['h5py'] = h5py
        print(f"✓ h5py {h5py.__version__} 导入成功")
    except Exception as e:
        print(f"✗ h5py 导入失败: {e}")
    
    return libraries

def inspect_mat_file(filepath, libraries):
    """详细检查MAT文件结构"""
    if 'h5py' not in libraries:
        return None, "h5py不可用"
    
    h5py = libraries['h5py']
    np = libraries['numpy']
    
    try:
        print(f"\n检查文件: {filepath}")
        print("="*60)
        
        with h5py.File(filepath, 'r') as f:
            print(f"文件根级别结构: {list(f.keys())}")
            
            if 'Sub' not in f:
                return None, "文件中没有找到'Sub'组"
            
            sub_group = f['Sub']
            print(f"\nSub组结构: {list(sub_group.keys())}")
            
            total_subjects = 0
            data_group_info = {}
            
            # 检查所有数据组
            for data_key in sub_group.keys():
                if data_key in ['events', 'meas_char', 'sub_char']:
                    print(f"\n{data_key} (元数据组):")
                    try:
                        meta_data = sub_group[data_key]
                        if hasattr(meta_data, 'shape'):
                            print(f"  形状: {meta_data.shape}")
                        elif hasattr(meta_data, 'keys'):
                            print(f"  包含: {list(meta_data.keys())}")
                    except Exception as e:
                        print(f"  无法读取: {e}")
                    continue
                
                print(f"\n数据组: {data_key}")
                data_array = sub_group[data_key]
                print(f"  数据组形状: {data_array.shape}")
                
                num_subjects = data_array.shape[0]
                data_group_info[data_key] = num_subjects
                total_subjects += num_subjects
                
                print(f"  受试者数量: {num_subjects}")
                
                # 检查前几个受试者的数据结构
                print(f"  检查前3个受试者的数据结构:")
                for i in range(min(3, num_subjects)):
                    try:
                        subject_ref = data_array[i, 0]
                        if isinstance(subject_ref, h5py.Reference):
                            subject_data = f[subject_ref]
                            if hasattr(subject_data, 'keys'):
                                body_parts = list(subject_data.keys())
                                print(f"    受试者 {i}: {len(body_parts)} 个身体部位")
                                print(f"      身体部位示例: {body_parts[:5]}...")
                            else:
                                print(f"    受试者 {i}: 直接数据，形状 {subject_data.shape}")
                        else:
                            print(f"    受试者 {i}: 非引用数据")
                    except Exception as e:
                        print(f"    受试者 {i}: 读取失败 - {e}")
            
            print(f"\n" + "="*60)
            print(f"总结:")
            print(f"  总数据组数量: {len(data_group_info)}")
            print(f"  总受试者数量: {total_subjects}")
            print(f"\n各数据组详情:")
            for group, count in data_group_info.items():
                print(f"  {group}: {count} 个受试者")
            
            return data_group_info, None
            
    except Exception as e:
        return None, f"检查文件失败: {e}"

def main():
    """主函数"""
    print("="*60)
    print("MAT文件结构详细检查")
    print("="*60)
    
    # 检查文件是否存在
    healthy_file = "../data/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat"
    stroke_file = "../data/MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"
    
    files_to_check = []
    if os.path.exists(healthy_file):
        files_to_check.append(("健康人数据", healthy_file))
    else:
        print(f"✗ 健康人数据文件不存在: {healthy_file}")
    
    if os.path.exists(stroke_file):
        files_to_check.append(("中风患者数据", stroke_file))
    else:
        print(f"✗ 中风患者数据文件不存在: {stroke_file}")
    
    if not files_to_check:
        print("没有找到MAT文件")
        return
    
    # 安装兼容的包
    install_compatible_packages()
    
    # 重新导入库
    print("\n重新导入库...")
    libraries = try_import_libraries()
    
    if not libraries or 'numpy' not in libraries:
        print("✗ 无法导入必要的库")
        return
    
    # 检查每个文件
    total_healthy = 0
    total_stroke = 0
    
    for file_desc, filepath in files_to_check:
        print(f"\n{'='*20} {file_desc} {'='*20}")
        data_info, error = inspect_mat_file(filepath, libraries)
        
        if data_info:
            file_total = sum(data_info.values())
            if "健康人" in file_desc:
                total_healthy = file_total
            else:
                total_stroke = file_total
            print(f"\n{file_desc}总计: {file_total} 个受试者")
        else:
            print(f"✗ {file_desc}检查失败: {error}")
    
    print(f"\n" + "="*60)
    print(f"最终统计:")
    print(f"  健康人总数: {total_healthy}")
    print(f"  中风患者总数: {total_stroke}")
    print(f"  总样本数: {total_healthy + total_stroke}")
    print("="*60)
    
    if total_healthy > 60 or total_stroke > 60:
        print("\n⚠️  发现问题:")
        print("   当前转换脚本可能没有提取所有数据!")
        print("   建议修改转换脚本以处理所有受试者数据")
    else:
        print("\n✓ 当前转换脚本已提取所有可用数据")

if __name__ == "__main__":
    main()
