#!/usr/bin/env python3
"""
Simple demonstration of the new Instagram options in the Gallery-dl node.
Shows how to use instagram_include and extra_options parameters.
"""

import os
import sys

# Add the nodes directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nodes'))

from gallery_dl_downloader import GalleryDLNode

def demonstrate_instagram_options():
    """Demonstrate the new Instagram options."""
    print("🎯 Gallery-dl Instagram Options Demo")
    print("=" * 50)
    
    # Create a node instance
    node = GalleryDLNode()
    
    # Get the input types to show available options
    input_types = node.INPUT_TYPES()
    
    print("\n📋 Available Instagram Include Options:")
    instagram_options = input_types["optional"]["instagram_include"][0]
    for i, option in enumerate(instagram_options, 1):
        print(f"   {i:2d}. {option}")
    
    print("\n🔧 Example Usage Scenarios:")
    
    scenarios = [
        {
            "name": "Stories Only",
            "url": "https://instagram.com/photographer",
            "instagram_include": "stories",
            "extra_options": "",
            "description": "Download only Instagram stories"
        },
        {
            "name": "Posts and Stories",
            "url": "https://instagram.com/artist", 
            "instagram_include": "posts,stories",
            "extra_options": "--range 1-10",
            "description": "Download posts and stories, limit to first 10 items"
        },
        {
            "name": "Everything Recent",
            "url": "https://instagram.com/creator",
            "instagram_include": "all",
            "extra_options": "--date-after 2024-01-01",
            "description": "Download all content types from 2024 onwards"
        },
        {
            "name": "High Quality Images",
            "url": "https://instagram.com/photographer",
            "instagram_include": "posts",
            "extra_options": '--filter "width >= 1080"',
            "description": "Download only high-resolution posts"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   URL: {scenario['url']}")
        print(f"   Instagram Include: {scenario['instagram_include']}")
        if scenario['extra_options']:
            print(f"   Extra Options: {scenario['extra_options']}")
        print(f"   Organize Files: True")
    
    print("\n💡 How to Use in ComfyUI:")
    print("1. Add 'Gallery-dl Downloader' node to your workflow")
    print("2. Enter Instagram URL in 'url_list' field")
    print("3. Select desired content type from 'instagram_include' dropdown")
    print("4. Add any extra gallery-dl options in 'extra_options' field")
    print("5. Enable 'organize_files' to sort downloads into folders")
    print("6. Set authentication (cookie_file or use_browser_cookies)")
    print("7. Run the workflow")
    
    print("\n🔒 Authentication Requirements:")
    print("- Instagram requires authentication for most content")
    print("- Use cookie_file: './configs/instagram_cookies.json'")
    print("- OR use_browser_cookies: True (with Firefox/Chrome)")
    print("- For stories/highlights: authentication is mandatory")
    print("- For public posts: may work without authentication")
    
    print("\n📁 File Organization:")
    print("When organize_files is enabled, downloads are sorted into:")
    print("- images/     - Photos and images")
    print("- videos/     - Video files")
    print("- audio/      - Audio files")
    print("- other/      - Other file types")
    
    print("\n⚠️ Important Notes:")
    print("- Don't include options in the URL field")
    print("- Use separate fields for instagram_include and extra_options")
    print("- Invalid extra_options will cause downloads to fail")
    print("- Rate limiting is automatically applied (1 second between requests)")
    
    print("\n🎉 Ready to Use!")
    print("Your Gallery-dl node now supports all Instagram content types!")

if __name__ == "__main__":
    demonstrate_instagram_options()
