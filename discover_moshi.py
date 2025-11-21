#!/usr/bin/env python3
"""
discover_moshi.py - Moshi Module Discovery
===========================================
Helps you figure out which Moshi command actually starts a WebSocket server.

Usage:
    python discover_moshi.py
"""

import sys
import subprocess
import importlib
import inspect
from pathlib import Path


def check_module_exists():
    """Check if moshi_mlx is installed."""
    try:
        import moshi_mlx
        print("✓ moshi_mlx is installed")
        print(f"  Location: {moshi_mlx.__file__}")
        return True
    except ImportError:
        print("✗ moshi_mlx not found")
        print("  Install with: pip install moshi_mlx")
        return False


def list_modules():
    """List all moshi_mlx modules."""
    try:
        import pkgutil
        import moshi_mlx
        
        print("\n" + "="*60)
        print("AVAILABLE MOSHI MODULES")
        print("="*60)
        
        modules = []
        for _, name, is_pkg in pkgutil.walk_packages(moshi_mlx.__path__):
            modules.append(name)
            pkg_str = " (package)" if is_pkg else ""
            print(f"  - {name}{pkg_str}")
        
        return modules
    except Exception as e:
        print(f"Error listing modules: {e}")
        return []


def try_module_help(module_name):
    """Try to get help for a module."""
    print(f"\n{'='*60}")
    print(f"MODULE: {module_name}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", f"moshi_mlx.{module_name}", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ Module has --help:")
            print(result.stdout[:500])
            if len(result.stdout) > 500:
                print("... (truncated)")
        else:
            print(f"⚠ Module returned error code {result.returncode}")
            if result.stderr:
                print("STDERR:", result.stderr[:200])
    
    except subprocess.TimeoutExpired:
        print("⚠ Module took too long (>5s)")
    except FileNotFoundError:
        print("✗ Cannot execute as module")
    except Exception as e:
        print(f"✗ Error: {e}")


def check_for_web_server(module_name):
    """Check if module mentions web/server in its code."""
    try:
        import moshi_mlx
        module_path = Path(moshi_mlx.__file__).parent / f"{module_name}.py"
        
        if not module_path.exists():
            return False
        
        content = module_path.read_text()
        
        web_keywords = ['websocket', 'WebSocket', 'server', 'localhost', '8998', 'web', 'http']
        found_keywords = [kw for kw in web_keywords if kw in content]
        
        if found_keywords:
            print(f"\n🌐 {module_name} mentions web-related keywords:")
            print(f"   Found: {', '.join(found_keywords)}")
            return True
        
        return False
    except Exception as e:
        return False


def main():
    print("\n" + "="*60)
    print("  MOSHI MODULE DISCOVERY TOOL")
    print("="*60 + "\n")
    
    # Check if moshi_mlx exists
    if not check_module_exists():
        return
    
    # List all modules
    modules = list_modules()
    
    if not modules:
        print("\nNo modules found!")
        return
    
    # Check each module for web server capabilities
    print("\n" + "="*60)
    print("CHECKING FOR WEB SERVER MODULES")
    print("="*60)
    
    web_modules = []
    for module in modules:
        if check_for_web_server(module):
            web_modules.append(module)
    
    # Get help for promising modules
    print("\n" + "="*60)
    print("MODULE DETAILS")
    print("="*60)
    
    for module in ['local', 'local_web', 'run_inference'] + web_modules:
        if module in modules:
            try_module_help(module)
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if 'local_web' in modules:
        print("\n1. Try running local_web and check what port it uses:")
        print("   python -m moshi_mlx.local_web")
        print("   Then check: lsof -i :8998")
        print()
    
    if 'local' in modules:
        print("2. The 'local' module exists (CLI mode)")
        print("   python -m moshi_mlx.local")
        print()
    
    print("3. Check moshi_mlx documentation:")
    print("   Try: python -c \"import moshi_mlx; help(moshi_mlx)\"")
    print()
    
    print("4. Look for a web/server script:")
    print(f"   ls -la $(python -c 'import moshi_mlx, os; print(os.path.dirname(moshi_mlx.__file__))')")
    print()
    
    if web_modules:
        print(f"5. Modules with web keywords: {', '.join(web_modules)}")
        print("   These might be your web server!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
    except Exception as e:
        print(f"\n[Fatal error: {e}]")
