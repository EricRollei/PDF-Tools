"""
Script to analyze the advanced sharpening processor details
"""
import os
import sys

# Path to the script
script_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes\scripts\advanced_sharpening.py"

print("=" * 80)
print("ANALYZING ADVANCED SHARPENING PROCESSOR")
print("=" * 80)
print()

if os.path.exists(script_path):
    print(f"✅ Found: {script_path}")
    print()
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📊 File size: {len(content)} characters")
    print(f"📊 Lines: {content.count(chr(10)) + 1}")
    print()
    
    # Find imports - check external dependencies
    print("📦 EXTERNAL DEPENDENCIES:")
    print("-" * 80)
    imports = []
    for line in content.split('\n'):
        if 'import ' in line and not line.strip().startswith('#'):
            imports.append(line.strip())
    
    # Categorize imports
    stdlib = []
    third_party = []
    local = []
    
    for imp in imports:
        if any(x in imp for x in ['torch', 'numpy', 'cv2', 'opencv', 'PIL', 'scipy', 'cupy', 'kornia']):
            third_party.append(imp)
        elif imp.startswith('from .') or imp.startswith('from ..'):
            local.append(imp)
        else:
            stdlib.append(imp)
    
    if third_party:
        print("  🔧 Third-party libraries:")
        for imp in third_party:
            print(f"     • {imp}")
    
    if local:
        print("  📁 Local imports:")
        for imp in local:
            print(f"     • {imp}")
    print()
    
    # Find class definitions
    print("🏛️  CLASSES:")
    print("-" * 80)
    for line in content.split('\n'):
        if line.strip().startswith('class '):
            class_name = line.split('class ')[1].split('(')[0].split(':')[0]
            print(f"  • {class_name}")
    print()
    
    # Find key methods
    print("⚙️  KEY METHODS:")
    print("-" * 80)
    methods_of_interest = [
        'smart_sharpen', 'auto_detect', 'adaptive', 'unsharp',
        'high_pass', 'laplacian', 'edge_aware', 'frequency',
        'process', 'sharpen', 'enhance'
    ]
    
    found_methods = []
    for line in content.split('\n'):
        if 'def ' in line:
            method_name = line.split('def ')[1].split('(')[0].strip()
            if any(keyword in method_name.lower() for keyword in methods_of_interest):
                found_methods.append(method_name)
    
    for method in found_methods[:20]:
        print(f"  • {method}")
    if len(found_methods) > 20:
        print(f"  ... and {len(found_methods) - 20} more")
    print()
    
    # Look for algorithm descriptions
    print("📝 ALGORITHM DESCRIPTIONS:")
    print("-" * 80)
    in_docstring = False
    docstring_lines = []
    
    for line in content.split('\n')[:100]:  # Check first 100 lines
        if '"""' in line or "'''" in line:
            if in_docstring:
                break
            else:
                in_docstring = True
                docstring_lines.append(line.replace('"""', '').replace("'''", ''))
        elif in_docstring:
            docstring_lines.append(line)
    
    if docstring_lines:
        for line in docstring_lines[:15]:
            if line.strip():
                print(f"  {line.strip()}")
        if len(docstring_lines) > 15:
            print(f"  ... (truncated)")
    print()
    
    # Check for GPU support
    print("🚀 GPU/PERFORMANCE FEATURES:")
    print("-" * 80)
    gpu_keywords = ['cuda', 'gpu', 'cupy', 'kornia', 'device']
    for keyword in gpu_keywords:
        if keyword.lower() in content.lower():
            print(f"  ✅ {keyword.upper()} support detected")
    print()
    
else:
    print(f"❌ File not found: {script_path}")

print()
print("=" * 80)

# Now check base_node to understand the interface
base_node_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes\base_node.py"
print("CHECKING BASE NODE INTERFACE")
print("=" * 80)
print()

if os.path.exists(base_node_path):
    print(f"✅ Found: {base_node_path}")
    with open(base_node_path, 'r', encoding='utf-8') as f:
        base_content = f.read()
    
    print(f"📊 File size: {len(base_content)} characters")
    print()
    
    # Check what BaseImageProcessingNode provides
    print("🏛️  BASE CLASS INTERFACE:")
    print("-" * 80)
    
    for line in base_content.split('\n'):
        if 'class BaseImageProcessingNode' in line:
            print(f"  {line.strip()}")
        if line.strip().startswith('def ') and 'self' in line:
            method = line.split('def ')[1].split('(')[0]
            print(f"  • {method}")
    print()
    
else:
    print(f"❌ Base node not found: {base_node_path}")
    
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
