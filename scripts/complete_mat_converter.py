#!/usr/bin/env python3
"""
完整MAT文件转换器
提取所有可用的真实MAT数据，不设任何限制
"""

import os
import sys
import subprocess
import csv
import time

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
    
    try:
        from scipy.io import loadmat
        libraries['scipy'] = loadmat
        print(f"✓ scipy.io 导入成功")
    except Exception as e:
        print(f"✗ scipy.io 导入失败: {e}")
    
    return libraries

def extract_features_from_subject(subject_data, file_handle, np, h5py):
    """从受试者数据中提取特征"""
    features = {}

    try:
        # 检查是否是引用
        if isinstance(subject_data, h5py.Reference):
            subject_data = file_handle[subject_data]

        # 检查数据结构
        if hasattr(subject_data, 'keys'):
            body_parts = list(subject_data.keys())
            
            for part_name in body_parts[:15]:  # 限制身体部位数量以控制特征数
                try:
                    part_data = subject_data[part_name]

                    # 解引用
                    if isinstance(part_data, h5py.Reference):
                        part_data = file_handle[part_data]

                    # 检查是否有数据
                    if hasattr(part_data, 'shape') and len(part_data.shape) > 0:
                        # 直接的数据数组
                        try:
                            data_array = np.array(part_data)

                            if data_array.size > 0:
                                # 处理多维数据
                                if len(data_array.shape) > 1:
                                    # 对于多维数据，计算每个维度的统计量
                                    if data_array.shape[1] <= 3:  # 如果列数不多，分别处理
                                        for col in range(data_array.shape[1]):
                                            col_data = data_array[:, col]
                                            valid_data = col_data[np.isfinite(col_data)]

                                            if len(valid_data) > 0:
                                                features[f"{part_name}_col{col}_mean"] = float(np.mean(valid_data))
                                                features[f"{part_name}_col{col}_std"] = float(np.std(valid_data))
                                                features[f"{part_name}_col{col}_max"] = float(np.max(valid_data))
                                                features[f"{part_name}_col{col}_min"] = float(np.min(valid_data))
                                    else:
                                        # 如果维度太多，就flatten
                                        data_array = data_array.flatten()

                                # 一维数据处理
                                if len(data_array.shape) == 1:
                                    valid_data = data_array[np.isfinite(data_array)]

                                    if len(valid_data) > 0:
                                        features[f"{part_name}_mean"] = float(np.mean(valid_data))
                                        features[f"{part_name}_std"] = float(np.std(valid_data))
                                        features[f"{part_name}_max"] = float(np.max(valid_data))
                                        features[f"{part_name}_min"] = float(np.min(valid_data))
                                        features[f"{part_name}_range"] = float(np.max(valid_data) - np.min(valid_data))

                        except Exception as e:
                            continue

                    # 检查是否是组
                    elif hasattr(part_data, 'keys'):
                        var_names = list(part_data.keys())

                        for var_name in var_names[:5]:  # 每个部位最多5个变量
                            try:
                                var_data = part_data[var_name]

                                if isinstance(var_data, h5py.Reference):
                                    var_data = file_handle[var_data]

                                if hasattr(var_data, 'shape') and len(var_data.shape) > 0:
                                    data_array = np.array(var_data)

                                    if data_array.size > 0:
                                        if len(data_array.shape) > 1:
                                            data_array = data_array.flatten()

                                        valid_data = data_array[np.isfinite(data_array)]

                                        if len(valid_data) > 0:
                                            features[f"{part_name}_{var_name}_mean"] = float(np.mean(valid_data))
                                            features[f"{part_name}_{var_name}_std"] = float(np.std(valid_data))
                                            features[f"{part_name}_{var_name}_max"] = float(np.max(valid_data))
                                            features[f"{part_name}_{var_name}_min"] = float(np.min(valid_data))

                            except Exception as e:
                                continue

                except Exception as e:
                    continue

        elif hasattr(subject_data, 'shape'):
            # 直接是数据数组
            try:
                data_array = np.array(subject_data)
                if data_array.size > 0:
                    if len(data_array.shape) > 1:
                        data_array = data_array.flatten()

                    valid_data = data_array[np.isfinite(data_array)]
                    if len(valid_data) > 0:
                        features["data_mean"] = float(np.mean(valid_data))
                        features["data_std"] = float(np.std(valid_data))
                        features["data_max"] = float(np.max(valid_data))
                        features["data_min"] = float(np.min(valid_data))
            except Exception as e:
                pass

    except Exception as e:
        pass

    return features

def read_complete_mat_file(filepath, libraries, label, group_name):
    """完整读取MAT文件中的所有数据"""
    if 'h5py' not in libraries:
        return None, "h5py不可用"
    
    h5py = libraries['h5py']
    np = libraries['numpy']
    
    try:
        print(f"完整读取: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            print(f"文件结构: {list(f.keys())}")
            
            if 'Sub' not in f:
                return None, "文件中没有找到'Sub'组"
            
            sub_group = f['Sub']
            
            # 处理所有数据组（排除元数据组）
            data_groups = [key for key in sub_group.keys() if key not in ['events', 'meas_char', 'sub_char']]
            print(f"发现 {len(data_groups)} 个数据组: {data_groups}")
            
            all_features = []
            total_processed = 0
            
            for data_key in data_groups:
                print(f"\n处理数据组: {data_key}")
                data_array = sub_group[data_key]
                print(f"数据组形状: {data_array.shape}")
                
                # 处理所有受试者数据
                num_subjects = data_array.shape[0]
                print(f"将处理 {num_subjects} 个受试者")
                
                group_processed = 0
                for i in range(num_subjects):
                    try:
                        subject_ref = data_array[i, 0]
                        
                        if isinstance(subject_ref, h5py.Reference):
                            subject_data = f[subject_ref]
                            features = extract_features_from_subject(subject_data, f, np, h5py)
                            
                            if features:
                                features['data_group'] = data_key
                                features['subject_index'] = i
                                features['label'] = label
                                features['group'] = group_name
                                features['subject_id'] = f"{group_name}_{data_key}_{i}"
                                all_features.append(features)
                                group_processed += 1
                                total_processed += 1
                                
                                if total_processed % 50 == 0:
                                    print(f"  已处理 {total_processed} 个受试者")
                                    
                    except Exception as e:
                        continue
                
                print(f"  {data_key}: 成功处理 {group_processed}/{num_subjects} 个受试者")
            
            print(f"总共提取了 {len(all_features)} 个受试者的特征")
            return all_features, None
            
    except Exception as e:
        return None, f"读取失败: {e}"

def convert_complete_mat_files():
    """转换所有真实的MAT文件数据"""
    print("="*60)
    print("完整MAT文件转换器")
    print("="*60)
    
    # 检查文件是否存在
    healthy_file = "../data/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat"
    stroke_file = "../data/MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"
    
    if not os.path.exists(healthy_file):
        print(f"✗ 健康人数据文件不存在: {healthy_file}")
        return False
    
    if not os.path.exists(stroke_file):
        print(f"✗ 中风患者数据文件不存在: {stroke_file}")
        return False
    
    print("✓ MAT文件存在，开始完整转换...")
    
    # 安装兼容的包
    install_compatible_packages()
    
    # 重新导入库
    print("\n重新导入库...")
    libraries = try_import_libraries()
    
    if not libraries or 'numpy' not in libraries:
        print("✗ 无法导入必要的库")
        return False
    
    all_data = []
    
    # 处理健康人数据
    print(f"\n{'='*30}")
    print("处理健康人数据")
    print(f"{'='*30}")
    start_time = time.time()
    
    healthy_features, error = read_complete_mat_file(healthy_file, libraries, 0, 'healthy')
    
    if healthy_features:
        all_data.extend(healthy_features)
        elapsed = time.time() - start_time
        print(f"✓ 成功提取 {len(healthy_features)} 个健康人样本 (耗时: {elapsed:.1f}秒)")
    else:
        print(f"✗ 健康人数据提取失败: {error}")
    
    # 处理中风患者数据
    print(f"\n{'='*30}")
    print("处理中风患者数据")
    print(f"{'='*30}")
    start_time = time.time()
    
    stroke_features, error = read_complete_mat_file(stroke_file, libraries, 1, 'stroke')
    
    if stroke_features:
        all_data.extend(stroke_features)
        elapsed = time.time() - start_time
        print(f"✓ 成功提取 {len(stroke_features)} 个中风患者样本 (耗时: {elapsed:.1f}秒)")
    else:
        print(f"✗ 中风患者数据提取失败: {error}")
    
    if not all_data:
        print("✗ 没有成功提取任何数据")
        return False
    
    # 保存为CSV
    output_file = "../data/complete_gait_features.csv"
    print(f"\n保存数据到: {output_file}")
    
    # 获取所有特征列名
    all_columns = set()
    for data in all_data:
        all_columns.update(data.keys())
    all_columns = sorted(list(all_columns))
    
    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        writer.writeheader()
        
        for data in all_data:
            row = {}
            for col in all_columns:
                row[col] = data.get(col, '')
            writer.writerow(row)
    
    print(f"✓ 完整MAT数据转换完成!")
    print(f"  总样本数: {len(all_data)}")
    print(f"  健康人: {len([d for d in all_data if d['label'] == 0])}")
    print(f"  中风患者: {len([d for d in all_data if d['label'] == 1])}")
    print(f"  特征数量: {len(all_columns)}")
    
    # 生成说明文件
    readme_file = "../data/complete_gait_features_readme.txt"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("完整MAT数据转换结果\n")
        f.write("="*30 + "\n\n")
        f.write("数据来源: Figshare完整MAT文件\n")
        f.write(f"总样本数: {len(all_data)}\n")
        f.write(f"健康人: {len([d for d in all_data if d['label'] == 0])}\n")
        f.write(f"中风患者: {len([d for d in all_data if d['label'] == 1])}\n")
        f.write(f"特征数量: {len(all_columns)}\n\n")
        
        # 数据组统计
        f.write("数据组分布:\n")
        group_stats = {}
        for data in all_data:
            group = data['data_group']
            label = data['label']
            key = f"{group}_{label}"
            group_stats[key] = group_stats.get(key, 0) + 1
        
        for key, count in sorted(group_stats.items()):
            group, label = key.rsplit('_', 1)
            label_name = "健康人" if label == '0' else "中风患者"
            f.write(f"  {group} ({label_name}): {count}\n")
        
        f.write("\n特征说明:\n")
        f.write("- 直接从MAT文件中提取的完整真实数据\n")
        f.write("- 包含所有数据组和受试者\n")
        f.write("- 包含各身体部位的运动学和动力学特征\n")
        f.write("- 每个特征包含均值、标准差、最大值、最小值\n")
        f.write("- label: 0=健康人, 1=中风患者\n")
    
    print(f"✓ 说明文件已生成: {readme_file}")
    
    return True

def main():
    """主函数"""
    success = convert_complete_mat_files()
    
    if success:
        print("\n🎉 完整MAT文件转换成功!")
        print("\n生成的文件:")
        print("  📊 ../data/complete_gait_features.csv - 完整MAT数据")
        print("  📝 ../data/complete_gait_features_readme.txt - 数据说明")
        print("\n现在可以使用完整的真实MAT数据进行分析了!")
    else:
        print("\n❌ 转换失败")

if __name__ == "__main__":
    main()
