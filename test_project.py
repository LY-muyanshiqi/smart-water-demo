"""
smart-water-demo 烟 smoke 测试
验证核心模块可导入且结构正确
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def test_src_directory_structure():
    """验证 src/ 目录结构"""
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    assert os.path.isdir(src_dir), "src/ 目录不存在"
    contents = os.listdir(src_dir)
    assert len(contents) > 0, "src/ 目录为空"


def test_requirements_file():
    """验证 requirements.txt 存在且非空"""
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    assert os.path.isfile(req_path), "requirements.txt 不存在"
    with open(req_path) as f:
        deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    assert len(deps) >= 5, f"依赖数量过少: {len(deps)}"
    for dep in deps:
        assert '>=' in dep or '==' in dep or '~=' in dep, f"依赖缺少版本约束: {dep}"


def test_docs_directory():
    """验证 docs/ 目录存在"""
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
    assert os.path.isdir(docs_dir), "docs/ 目录不存在"


def test_readme_exists():
    """验证 README.md 存在"""
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
    assert os.path.isfile(readme), "README.md 不存在"
    with open(readme, encoding='utf-8') as f:
        content = f.read()
    assert 'smart-water-demo' in content.lower()
