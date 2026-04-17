import os
import subprocess
import sys

def get_git_version():
    try:
        # 获取最近的 tag 或 commit hash
        tag = subprocess.check_output(['git', 'describe', '--tags', '--dirty', '--always'], stderr=subprocess.DEVNULL).decode().strip()
        return tag
    except:
        return "0.0.0"

def write_version_file(version):
    version_file = os.path.join(os.path.dirname(__file__), 'version.py')
    with open(version_file, 'w') as f:
        f.write(f'__version__ = "{version}"\n')
    print(f"Written version {version} to {version_file}")

if __name__ == '__main__':
    version = get_git_version()
    write_version_file(version)
    sys.exit(0)