#!/usr/bin/env python3
"""Check memory backend dependencies"""

def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name} available")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} not available: {e}")
        return False

print("Checking memory backend dependencies...")
print("=" * 50)

# Core dependencies
check_import("redis", "Redis")
check_import("chromadb", "ChromaDB")
check_import("sentence_transformers", "Sentence Transformers")
check_import("mem0", "Mem0")

print("\nChecking optional dependencies...")
check_import("numpy", "NumPy")
check_import("pandas", "Pandas")