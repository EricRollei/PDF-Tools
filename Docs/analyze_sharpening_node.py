"""
Script to analyze the advanced sharpening node structure
"""
import os
import sys

# Path to the other node set
image_processing_path = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_Image_Processing_Nodes"
sharpening_node_path = os.path.join(image_processing_path, "nodes", "advanced_sharpening_node.py")

print("=" * 80)
print("ANALYZING ADVANCED SHARPENING NODE")
print("=" * 80)

if os.path.exists(sharpening_node_path):
    print(f"✅ Found: {sharpening_node_path}")
    print()
    
    # Read the file
    with open(sharpening_node_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic analysis
    print(f"📊 File size: {len(content)} characters")
    print(f"📊 Lines: {content.count(chr(10)) + 1}")
    print()
    
    # Find imports
    print("📦 IMPORTS:")
    print("-" * 80)
    for line in content.split('\n'):
        if 'import ' in line and not line.strip().startswith('#'):
            print(f"  {line.strip()}")
    print()
    
    # Find class definitions
    print("🏛️  CLASS DEFINITIONS:")
    print("-" * 80)
    for line in content.split('\n'):
        if line.strip().startswith('class '):
            print(f"  {line.strip()}")
    print()
    
    # Find function definitions
    print("⚙️  FUNCTION DEFINITIONS:")
    print("-" * 80)
    function_count = 0
    for line in content.split('\n'):
        if line.strip().startswith('def '):
            # Extract function name
            func_name = line.split('def ')[1].split('(')[0]
            print(f"  • {func_name}")
            function_count += 1
            if function_count > 30:
                print(f"  ... and more (truncated)")
                break
    print()
    
    # Check for external dependencies
    print("🔗 EXTERNAL DEPENDENCIES:")
    print("-" * 80)
    scripts_dir = os.path.join(image_processing_path, "scripts")
    if os.path.exists(scripts_dir):
        print(f"  ✅ Scripts directory exists: {scripts_dir}")
        scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
        print(f"  📁 Found {len(scripts)} Python script(s):")
        for script in scripts[:10]:
            print(f"     • {script}")
        if len(scripts) > 10:
            print(f"     ... and {len(scripts) - 10} more")
    else:
        print(f"  ❌ No scripts directory found")
    print()
    
    # Look for specific sharpening methods
    print("🔍 SHARPENING METHODS DETECTED:")
    print("-" * 80)
    keywords = [
        'unsharp', 'smart_sharpen', 'auto_detect', 'adaptive', 
        'frequency', 'wavelet', 'laplacian', 'bilateral',
        'gaussian', 'edge_enhance'
    ]
    
    for keyword in keywords:
        if keyword.lower() in content.lower():
            # Find the line
            for line in content.split('\n'):
                if keyword.lower() in line.lower() and ('def ' in line or 'class ' in line):
                    print(f"  ✅ {keyword}: {line.strip()[:70]}")
                    break
    print()
    
    # Check for INPUT_TYPES
    print("🎛️  NODE CONFIGURATION:")
    print("-" * 80)
    if 'INPUT_TYPES' in content:
        print("  ✅ Has INPUT_TYPES method (ComfyUI node)")
        # Extract some details
        input_start = content.find('INPUT_TYPES')
        input_section = content[input_start:input_start+1000]
        if 'sharpening_method' in input_section or 'method' in input_section:
            print("  ✅ Has method/mode selection")
        if 'strength' in input_section or 'amount' in input_section:
            print("  ✅ Has strength/amount parameter")
    else:
        print("  ❌ No INPUT_TYPES found (may not be a ComfyUI node)")
    print()
    
    # Check for imports from scripts
    print("📂 SCRIPT IMPORTS:")
    print("-" * 80)
    import_lines = [line for line in content.split('\n') if 'from' in line and 'scripts' in line]
    if import_lines:
        for line in import_lines[:10]:
            print(f"  {line.strip()}")
    else:
        print("  ℹ️  No script imports detected")
    print()
    
else:
    print(f"❌ File not found: {sharpening_node_path}")
    print()
    print("Checking if directory exists...")
    if os.path.exists(image_processing_path):
        print(f"✅ Directory exists: {image_processing_path}")
        print(f"📁 Contents:")
        for item in os.listdir(image_processing_path)[:20]:
            item_path = os.path.join(image_processing_path, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
    else:
        print(f"❌ Directory not found: {image_processing_path}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
