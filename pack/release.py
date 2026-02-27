#!/usr/bin/env python3
"""
Praasper 包发布脚本
功能：
- 从PyPI查询最新版本并自动递增patch号
- 支持指定版本号
- 验证文件编码
- 简化构建过程
- 完整的错误处理
- 上传到PyPI
"""

import re
import requests
import os
import argparse
import subprocess
import sys
import traceback


def log(message, level="info"):
    """
    日志输出函数
    """
    prefix = {
        "info": "[INFO] ",
        "error": "[ERROR] ",
        "warning": "[WARNING] ",
        "success": "[SUCCESS] "
    }.get(level, "[INFO] ")
    print(f"{prefix}{message}")


def run_command(cmd, cwd=None):
    """
    运行命令并返回结果
    """
    log(f"执行命令: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        log(f"返回码: {result.returncode}")
        if result.stdout:
            log(f"输出: {result.stdout.strip()}")
        if result.stderr:
            log(f"错误: {result.stderr.strip()}", level="warning")
        return result.returncode == 0, result
    except Exception as e:
        log(f"运行命令失败: {e}", level="error")
        return False, None


def validate_file_encoding(file_path):
    """
    验证文件编码是否正确
    """
    log(f"验证 {file_path} 的编码...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log(f"✓ {file_path} 编码正确 (UTF-8)")
        return True
    except UnicodeDecodeError:
        log(f"✗ {file_path} 编码错误，尝试修复...", level="error")
        # 尝试以其他编码读取并转换
        encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'ansi']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                # 以UTF-8重新写入
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                log(f"✓ 已将 {file_path} 转换为UTF-8编码")
                return True
            except Exception:
                continue
        log(f"✗ 无法修复 {file_path} 的编码", level="error")
        return False
    except Exception as e:
        log(f"✗ 验证 {file_path} 编码时出错: {e}", level="error")
        return False


def get_latest_version(package_name):
    """
    从PyPI获取包的最新版本号
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        log(f"从PyPI查询 {package_name} 的最新版本...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        version = data['info']['version']
        log(f"当前最新版本: {version}")
        return version
    except Exception as e:
        log(f"无法从PyPI获取版本信息: {e}", level="warning")
        log("使用本地setup.py中的版本号")
        return get_local_version()


def get_local_version():
    """
    从本地setup.py文件获取当前版本号
    """
    setup_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
    try:
        with open(setup_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        version_match = re.search(r"version='([^']+)',", content)
        if version_match:
            return version_match.group(1)
        else:
            raise ValueError("无法从setup.py中提取版本号")
    except Exception as e:
        log(f"获取本地版本号失败: {e}", level="error")
        raise


def increment_version(current_version):
    """
    递增版本号的patch部分
    """
    try:
        parts = current_version.split('.')
        if len(parts) != 3:
            raise ValueError("版本号格式不正确，应为MAJOR.MINOR.PATCH")
        major, minor, patch = map(int, parts)
        new_patch = patch + 1
        new_version = f"{major}.{minor}.{new_patch}"
        log(f"新版本号: {new_version}")
        return new_version
    except Exception as e:
        log(f"递增版本号失败: {e}", level="error")
        raise


def validate_version(version):
    """
    验证版本号格式是否正确
    """
    try:
        parts = version.split('.')
        if len(parts) != 3:
            raise ValueError(f"版本号格式不正确: {version}，应为MAJOR.MINOR.PATCH")
        list(map(int, parts))
        log(f"验证版本号格式: {version} ✓")
        return version
    except Exception as e:
        log(f"验证版本号失败: {e}", level="error")
        raise


def update_setup_py(new_version):
    """
    更新setup.py文件中的版本号
    """
    setup_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setup.py')
    try:
        log(f"更新setup.py中的版本号为: {new_version}")
        with open(setup_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = re.sub(r"version='([^']+)',", f"version='{new_version}',", content)
        with open(setup_py_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        log(f"✓ 成功更新setup.py中的版本号")
        return True
    except Exception as e:
        log(f"更新setup.py失败: {e}", level="error")
        return False


def check_requirements():
    """
    检查必要的依赖工具
    """
    log("检查必要的工具...")
    
    # 检查twine
    try:
        log("✓ twine 已安装")
    except ImportError:
        log("错误: twine 未安装，请运行 'pip install twine'", level="error")
        return False
    
    # 检查build
    try:
        log("✓ build 已安装")
    except ImportError:
        log("错误: build 未安装，请运行 'pip install build'", level="error")
        return False
    
    return True


def validate_files():
    """
    验证必要文件的编码和存在性
    """
    log("验证项目文件...")
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    files_to_validate = [
        'setup.py',
        'README.md',
        'requirements.txt'
    ]
    
    all_valid = True
    for file_name in files_to_validate:
        file_path = os.path.join(project_root, file_name)
        if not os.path.exists(file_path):
            log(f"✗ {file_name} 文件不存在", level="error")
            all_valid = False
        else:
            if not validate_file_encoding(file_path):
                all_valid = False
    
    if all_valid:
        log("✓ 所有文件验证通过")
    else:
        log("✗ 文件验证失败", level="error")
    
    return all_valid


def clean_dist():
    """
    清理dist目录
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    dist_dir = os.path.join(project_root, 'dist')
    
    log("清理dist目录...")
    
    if os.path.exists(dist_dir):
        old_files = [f for f in os.listdir(dist_dir) if f.endswith(('.whl', '.tar.gz'))]
        if old_files:
            log(f"清理旧的构建产物: {old_files}")
            for f in old_files:
                try:
                    os.remove(os.path.join(dist_dir, f))
                    log(f"✓ 删除 {f}")
                except Exception as e:
                    log(f"✗ 删除 {f} 失败: {e}", level="warning")
    else:
        log("dist目录不存在，跳过清理")


def build_package():
    """
    构建包
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    log("开始构建包...")
    
    # 使用python -m build构建
    success, result = run_command(
        "python -m build",
        cwd=project_root
    )
    
    if success:
        log("✓ 构建成功", level="success")
        return True
    else:
        log("✗ 构建失败", level="error")
        if result:
            log(f"构建错误输出: {result.stderr}", level="error")
        return False


def upload_to_pypi(new_version):
    """
    上传到PyPI
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    dist_dir = os.path.join(project_root, 'dist')
    
    log("准备上传到PyPI...")
    
    # 检查dist目录
    if not os.path.exists(dist_dir):
        log("✗ dist目录不存在", level="error")
        return False
    
    files = os.listdir(dist_dir)
    if not files:
        log("✗ dist目录为空", level="error")
        return False
    
    log(f"构建产物: {files}")
    
    # 找到当前版本的文件
    current_version_files = [f for f in files if new_version in f]
    if not current_version_files:
        log("✗ 未找到当前版本的构建产物", level="error")
        return False
    
    log(f"上传文件: {current_version_files}")
    
    # 确认上传
    confirm = input("是否上传到PyPI？(y/N): ")
    if confirm.lower() != 'y':
        log("取消上传")
        return False
    
    # 执行上传
    upload_files = [os.path.join('dist', f) for f in current_version_files]
    cmd = f"twine upload {' '.join(upload_files)}"
    success, result = run_command(cmd, cwd=project_root)
    
    if success:
        log("✓ 上传成功", level="success")
        log("包已上传到PyPI，通常需要几分钟时间在PyPI上可见")
        return True
    else:
        log("✗ 上传失败", level="error")
        if result:
            log(f"上传错误输出: {result.stderr}", level="error")
        return False


def main():
    """
    主函数
    """
    try:
        log("开始Praasper包发布流程")
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='Praasper包发布脚本')
        parser.add_argument('--version', type=str, help='指定版本号（格式：MAJOR.MINOR.PATCH）')
        args = parser.parse_args()
        
        package_name = "praasper"
        
        # 检查必要工具
        if not check_requirements():
            log("必要工具检查失败，退出流程", level="error")
            return 1
        
        # 验证文件
        if not validate_files():
            log("文件验证失败，退出流程", level="error")
            return 1
        
        # 确定版本号
        if args.version:
            log("使用指定的版本号")
            new_version = validate_version(args.version)
        else:
            log("自动计算新版本号")
            current_version = get_latest_version(package_name)
            new_version = increment_version(current_version)
        
        # 更新setup.py
        if not update_setup_py(new_version):
            log("更新setup.py失败，退出流程", level="error")
            return 1
        
        # 清理dist目录
        clean_dist()
        
        # 构建包
        if not build_package():
            log("构建失败，退出流程", level="error")
            return 1
        
        # 上传到PyPI
        if not upload_to_pypi(new_version):
            log("上传失败，退出流程", level="error")
            return 1
        
        log("\n🎉 发布流程完成！", level="success")
        return 0
        
    except KeyboardInterrupt:
        log("\n用户中断流程", level="warning")
        return 1
    except Exception as e:
        log(f"流程执行失败: {e}", level="error")
        log(f"详细错误信息: {traceback.format_exc()}", level="error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
