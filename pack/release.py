#!/usr/bin/env python3
"""
自动更新版本号脚本
功能：
- 默认：查询PyPI上的最新版本，自动递增patch号
- 支持通过--version参数指定版本号
"""

import re
import requests
import os
import argparse
import subprocess


def run_command(cmd):
    """
    运行命令并返回结果
    """
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"返回码: {result.returncode}")
    if result.stdout:
        print(f"输出: {result.stdout}")
    if result.stderr:
        print(f"错误: {result.stderr}")
    return result.returncode == 0


def get_latest_version(package_name):
    """
    从PyPI获取包的最新版本号
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['info']['version']
    except Exception as e:
        print(f"警告：无法从PyPI获取版本信息: {e}")
        print("使用本地setup.py中的版本号")
        return get_local_version()


def get_local_version():
    """
    从本地setup.py文件获取当前版本号
    """
    # 脚本在release文件夹中，需要向上一级目录找setup.py
    setup_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    version_match = re.search(r"version='([^']+)',", content)
    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("无法从setup.py中提取版本号")


def increment_version(current_version):
    """
    递增版本号的patch部分
    """
    parts = current_version.split('.')
    if len(parts) != 3:
        raise ValueError("版本号格式不正确，应为MAJOR.MINOR.PATCH")
    
    major, minor, patch = map(int, parts)
    new_patch = patch + 1
    
    return f"{major}.{minor}.{new_patch}"


def validate_version(version):
    """
    验证版本号格式是否正确
    """
    parts = version.split('.')
    if len(parts) != 3:
        raise ValueError(f"版本号格式不正确: {version}，应为MAJOR.MINOR.PATCH")
    try:
        list(map(int, parts))
    except ValueError:
        raise ValueError(f"版本号格式不正确: {version}，应为数字格式")
    return version


def update_setup_py(new_version):
    """
    更新setup.py文件中的版本号
    """
    # 脚本在release文件夹中，需要向上一级目录找setup.py
    setup_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
    
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = re.sub(r"version='([^']+)',", f"version='{new_version}',", content)
    
    with open(setup_py_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已更新setup.py中的版本号为: {new_version}")


def check_requirements():
    """
    检查必要的依赖工具
    """
    print("检查必要的工具...")
    
    # 检查twine
    try:
        import twine
        print("✓ twine 已安装")
    except ImportError:
        print("错误: twine 未安装，请运行 'pip install twine'")
        return False
    
    # 检查build
    try:
        import build
        print("✓ build 已安装")
    except ImportError:
        print("错误: build 未安装，请运行 'pip install build'")
        return False
    
    return True


def main():
    """
    主函数
    """
    # 检查必要工具
    if not check_requirements():
        return
    
    parser = argparse.ArgumentParser(description='自动更新版本号脚本')
    parser.add_argument('--version', type=str, help='指定版本号（格式：MAJOR.MINOR.PATCH）')
    args = parser.parse_args()
    
    package_name = "praasper"
    
    if args.version:
        # 使用指定的版本号
        print("步骤1: 使用指定的版本号...")
        new_version = validate_version(args.version)
        print(f"指定版本号: {new_version}")
    else:
        # 默认行为：查询最新版本并递增patch号
        print("步骤1: 查询最新版本...")
        current_version = get_latest_version(package_name)
        print(f"当前最新版本: {current_version}")
        
        print("步骤2: 计算新版本号...")
        new_version = increment_version(current_version)
        print(f"新版本号: {new_version}")
    
    print("步骤3: 更新setup.py文件...")
    update_setup_py(new_version)
    
    print("\n步骤4: 执行发布流程...")
    
    # 脚本在release文件夹中，dist目录在项目根目录
    dist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist')
    
    # 清理dist目录中的旧版本
    print("\n执行步骤0: 清理dist目录中的旧版本")
    if os.path.exists(dist_dir):
        old_files = [f for f in os.listdir(dist_dir) if f.endswith(('.whl', '.tar.gz'))]
        if old_files:
            print(f"清理旧的构建产物: {old_files}")
            for f in old_files:
                os.remove(os.path.join(dist_dir, f))
    
    # 1. 清理旧构建
    print("\n执行步骤1: 清理旧构建")
    # 构建目录在项目根目录
    build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
    if os.path.exists(build_dir):
        # 切换到项目根目录执行清理命令
        project_root = os.path.dirname(os.path.dirname(__file__))
        if not run_command(f"cd {project_root} && python setup.py clean --all"):
            print("警告: 清理构建失败，但继续执行")
    else:
        print("构建目录不存在，跳过清理步骤")
    
    # 2. 构建包
    print("\n执行步骤2: 构建包")
    # 在项目根目录执行构建命令
    project_root = os.path.dirname(os.path.dirname(__file__))
    if not run_command(f"cd {project_root} && python -m build"):
        print("错误: 构建失败，终止流程")
        return
    
    # 3. 上传到PyPI
    print("\n执行步骤3: 上传到PyPI")
    print("注意: 这将上传到正式的PyPI，请确保版本号正确！")
    
    # 检查dist目录是否存在构建产物
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        print(f"构建产物: {files}")
        if not files:
            print("错误: 构建目录为空，终止上传")
            return
    else:
        print("错误: 构建目录不存在，终止上传")
        return
    
    # 添加确认步骤
    confirm = input("是否继续上传？(y/N): ")
    if confirm.lower() != 'y':
        print("取消上传，构建产物保留在dist目录中")
        return
    
    # 执行上传 - 只上传当前版本的包
    current_version_files = [f for f in files if f"{new_version}" in f]
    if current_version_files:
        print(f"只上传当前版本的包: {current_version_files}")
        # 在项目根目录执行上传命令
        project_root = os.path.dirname(os.path.dirname(__file__))
        upload_files = [f"dist/{f}" for f in current_version_files]
        upload_cmd = f"cd {project_root} && twine upload {' '.join(upload_files)}"
        if run_command(upload_cmd):
            print("\n🎉 发布成功！")
            print("包已上传到PyPI，通常需要几分钟时间在PyPI上可见")
        else:
            print("\n错误: 上传失败")
    else:
        print("错误: 未找到当前版本的构建产物，终止上传")
        return
    
    print("\n版本更新和发布流程完成！")


if __name__ == "__main__":
    main()
