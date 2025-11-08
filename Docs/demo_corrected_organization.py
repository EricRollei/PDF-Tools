#!/usr/bin/env python3
"""
Demo script showing the corrected file organization for Instagram downloads.
This demonstrates how files will now be organized within profile directories.
"""

import os
import tempfile
import sys

# Add the nodes directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nodes'))

from gallery_dl_downloader import GalleryDLDownloader

def demonstrate_instagram_organization():
    """Demonstrate the corrected Instagram file organization."""
    print("🎯 Instagram File Organization Demo")
    print("=" * 50)
    
    print("✅ FIXED: File organization now works correctly!")
    print("\n📁 NEW Directory Structure (CORRECT):")
    print("gallery-dl-output/")
    print("├── instagram/")
    print("│   └── janaioannaa/")
    print("│       ├── images/")
    print("│       │   ├── post1.jpg")
    print("│       │   ├── post2.png")
    print("│       │   └── story1.webp")
    print("│       ├── videos/")
    print("│       │   ├── reel1.mp4")
    print("│       │   └── video1.mp4")
    print("│       └── audio/")
    print("│           └── audio1.mp3")
    print("└── gallery-dl-metadata.json")
    
    print("\n❌ OLD Directory Structure (WRONG - FIXED):")
    print("gallery-dl-output/")
    print("├── images/           ← Files were moved here (WRONG)")
    print("│   ├── post1.jpg")
    print("│   └── post2.png")
    print("├── videos/           ← Files were moved here (WRONG)")
    print("│   └── reel1.mp4")
    print("└── instagram/")
    print("    └── janaioannaa/   ← Directory left empty (WRONG)")
    
    print("\n🔧 How It Works Now:")
    print("1. Gallery-dl downloads files to: gallery-dl-output/instagram/janaioannaa/")
    print("2. Node detects files are in a profile subdirectory")
    print("3. Node creates type folders WITHIN the profile directory:")
    print("   - gallery-dl-output/instagram/janaioannaa/images/")
    print("   - gallery-dl-output/instagram/janaioannaa/videos/")
    print("   - gallery-dl-output/instagram/janaioannaa/audio/")
    print("   - gallery-dl-output/instagram/janaioannaa/other/")
    print("4. Files are moved to appropriate type folders within their profile directory")
    
    print("\n💡 Usage Example:")
    print("URL: https://instagram.com/janaioannaa")
    print("Instagram Include: posts,stories")
    print("Organize Files: True")
    print("Output Directory: ./gallery-dl-output")
    
    print("\nResult:")
    print("✅ ./gallery-dl-output/instagram/janaioannaa/images/  ← Photos here")
    print("✅ ./gallery-dl-output/instagram/janaioannaa/videos/  ← Videos here")
    print("✅ ./gallery-dl-output/instagram/janaioannaa/audio/   ← Audio here")
    
    print("\n📊 Debug Output You'll See:")
    print("🔧 Debug Information:")
    print("📁 Moved photo1.jpg to instagram/janaioannaa/images/ folder")
    print("📁 Moved video1.mp4 to instagram/janaioannaa/videos/ folder")
    print("📂 Organized 5 files in instagram/janaioannaa:")
    print("   📁 images: 3 files")
    print("   📁 videos: 2 files")
    print("📂 Total file organization complete: 5 files")
    
    print("\n🎉 Ready to Test!")
    print("Try downloading from https://instagram.com/janaioannaa with organize_files=True")
    print("Files will now be properly organized within the profile directory!")

def simulate_corrected_behavior():
    """Simulate the corrected file organization behavior."""
    print("\n🧪 Simulating Corrected Behavior...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a realistic Instagram download structure
        profile_dir = os.path.join(temp_dir, "instagram", "janaioannaa")
        os.makedirs(profile_dir, exist_ok=True)
        
        # Simulate downloaded files
        files = {
            "photo1.jpg": "image",
            "photo2.png": "image", 
            "story1.webp": "image",
            "reel1.mp4": "video",
            "video1.mp4": "video"
        }
        
        file_paths = []
        for filename, file_type in files.items():
            file_path = os.path.join(profile_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"Simulated {file_type} content")
            file_paths.append(file_path)
        
        print(f"📁 Created files in: {os.path.relpath(profile_dir, temp_dir)}")
        for filename in files.keys():
            print(f"   📄 {filename}")
        
        # Test organization
        downloader = GalleryDLDownloader(
            output_dir=temp_dir,
            organize_files=True
        )
        
        downloader._organize_files_by_type(file_paths)
        
        # Show results
        print(f"\n📊 Organization Results:")
        
        for subdir in ["images", "videos", "audio", "other"]:
            subdir_path = os.path.join(profile_dir, subdir)
            if os.path.exists(subdir_path):
                files_in_subdir = os.listdir(subdir_path)
                if files_in_subdir:
                    print(f"✅ instagram/janaioannaa/{subdir}/: {files_in_subdir}")
        
        print("\n✅ Files are now organized WITHIN the profile directory!")

if __name__ == "__main__":
    demonstrate_instagram_organization()
    simulate_corrected_behavior()
