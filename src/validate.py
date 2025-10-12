"""
Pre-flight validation script to check environment setup
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    required = [
        'pandas', 'numpy', 'sklearn', 'lightgbm',
        'transformers', 'sentence_transformers', 'faiss',
        'torch', 'PIL', 'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """Check if data files exist"""
    files = [
        'dataset/train.csv',
        'dataset/test.csv'
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    return all_exist

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024 ** 3)
        
        if free_gb < 10:
            print(f"⚠️  Low disk space: {free_gb:.1f} GB free (10+ GB recommended)")
            return False
        else:
            print(f"✅ Disk space: {free_gb:.1f} GB free")
            return True
    except:
        print("⚠️  Could not check disk space")
        return True

def check_memory():
    """Check available RAM"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        
        if total_gb < 12:
            print(f"⚠️  Low RAM: {total_gb:.1f} GB total (16+ GB recommended)")
            print(f"   Available: {available_gb:.1f} GB")
            return False
        else:
            print(f"✅ RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
            return True
    except ImportError:
        print("⚠️  Could not check RAM (psutil not installed)")
        return True

def create_directories():
    """Create necessary directories"""
    dirs = ['cache', 'images', 'images/train', 'images/test']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Directory: {dir_name}")
    
    return True

def main():
    """Run all checks"""
    print("=" * 60)
    print("PRE-FLIGHT VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Disk Space", check_disk_space),
        ("Memory", check_memory),
        ("Directories", create_directories)
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! Ready to run pipeline.")
        print("\nRun: python train.py")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before running.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
