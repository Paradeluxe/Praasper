#!/usr/bin/env python3
"""
完整的发布脚本
功能：自动更新版本号 + 构建包 + 上传到PyPI
"""

import os
import sys
import subprocess


def run_command(cmd, cwd=None):
    """
    运行命令并返回结果
    """
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    print(f"返回码: {result.returncode}")
    if result.stdout:
        print(f"输出: {result.stdout}")
    if result.stderr:
        print(f"错误: {result.stderr}")
    return result


def main():
    """
    主发布流程
    """
    print("=== Praasper 发布流程 ===")
    
    # 步骤1: 更新版本号
    print("\n步骤1: 更新版本号")
    version_result = run_command("python update_version.py")
    if version_result.returncode != 0:
        print("错误: 版本更新失败")
        sys.exit(1)
    
    # 步骤2: 清理旧构建
    print("\n步骤2: 清理旧构建")
    clean_result = run_command("python setup.py clean --all")
    if clean_result.returncode != 0:
        print("警告: 清理构建失败，但继续执行")
    
    # 步骤3: 构建包
    print("\n步骤3: 构建包")
    build_result = run_command("python setup.py sdist bdist_wheel")
    if build_result.returncode != 0:
        print("错误: 构建失败")
        sys.exit(1)
    
    # 步骤4: 检查构建结果
    print("\n步骤4: 检查构建结果")
    dist_dir = "dist"
    if os.path.exists(dist_dir):
        files = os.listdir(dist_dir)
        print(f"构建产物: {files}")
        if not files:
            print("错误: 构建目录为空")
            sys.exit(1)
    else:
        print("错误: 构建目录不存在")
        sys.exit(1)
    
    # 步骤5: 上传到PyPI
    print("\n步骤5: 上传到PyPI")
    print("注意: 这将上传到正式的PyPI，请确保版本号正确！")
    confirm = input("是否继续上传？(y/N): ")
    
    if confirm.lower() == 'y':
        upload_result = run_command("twine upload dist/*")
        if upload_result.returncode != 0:
            print("错误: 上传失败")
            sys.exit(1)
        print("\n🎉 发布成功！")
        print("包已上传到PyPI，通常需要几分钟时间在PyPI上可见")
    else:
        print("取消上传，构建产物保留在dist目录中")
        print("如果需要手动上传，可执行命令: twine upload dist/*")
    
    print("\n=== 发布流程完成 ===")


if __name__ == "__main__":
    main()
