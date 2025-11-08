# Gallery-dl Advanced Options Guide

## 🆕 New Advanced Features

The Gallery-dl ComfyUI node now supports advanced gallery-dl options including Instagram-specific parameters and custom command-line options.

## 📸 Instagram Advanced Options

### Instagram Include Parameter
Control what content to download from Instagram profiles using the `instagram_include` parameter:

#### Available Options:
- **posts** (default) - Regular posts only
- **stories** - Stories only
- **highlights** - Highlights only
- **reels** - Reels only
- **tagged** - Posts where user is tagged
- **info** - Profile information
- **avatar** - Profile avatar
- **all** - Everything
- **posts,stories** - Combination of posts and stories
- **stories,highlights** - Combination of stories and highlights

#### Example Usage:
```
URL: https://instagram.com/username
instagram_include: stories
```

This will download only the stories from the Instagram profile.

### Common Instagram Combinations:
- `posts,stories` - Get both posts and stories
- `posts,reels` - Get posts and reels
- `stories,highlights` - Get stories and highlights
- `all` - Get everything (posts, stories, highlights, reels, tagged, info, avatar)

## 🔧 Extra Options

Use the `extra_options` field to add any gallery-dl command-line options:

### Range Downloads
Download only a specific range of items:
```
extra_options: --range 1-10
```

### File Filtering
Download only specific file types:
```
extra_options: --filter "extension in ('jpg', 'png')"
```

### Custom Sleep/Retry Settings
Override default rate limiting:
```
extra_options: --sleep 2.0 --retries 5
```

### Skip Metadata
Don't save metadata files:
```
extra_options: --no-mtime --no-part
```

### Date Filtering
Download only recent items:
```
extra_options: --date-after 2024-01-01
```

## 🎯 Complete Example Usage

### Instagram Stories with Range
```
URL: https://instagram.com/photographer
instagram_include: stories
extra_options: --range 1-5
organize_files: True
```

### Instagram Posts with Custom Filter
```
URL: https://instagram.com/artist
instagram_include: posts,reels
extra_options: --filter "width >= 1080"
organize_files: True
```

### Multi-type Download with Date Range
```
URL: https://instagram.com/creator
instagram_include: all
extra_options: --date-after 2024-06-01 --range 1-20
organize_files: True
```

## 🚫 What WON'T Work

### ❌ Wrong URL Format
```
# This is WRONG - don't include options in URL
URL: https://instagram.com/username -o stories
```

### ✅ Correct Format
```
# This is CORRECT - use separate fields
URL: https://instagram.com/username
instagram_include: stories
```

## 🔍 Debug Information

The node provides detailed debug information showing:
- Which Instagram include options are being used
- What extra options are being added
- Dynamic config file creation
- Command-line being executed

Example debug output:
```
🔧 Debug Information:
🎯 Detected target sites: instagram
📸 Instagram include: stories,highlights
📄 Created dynamic config with 1 sections
📄 Using dynamic config: /tmp/dynamic_config.json
➕ Added extra options: --range 1-10 --no-mtime
⚡ Rate limiting: 1.0 second between requests, 3 retries on failure
```

## 📚 Gallery-dl Documentation Reference

For more advanced options, refer to the official gallery-dl documentation:
- [Configuration](https://github.com/mikf/gallery-dl/blob/master/docs/configuration.rst)
- [Command-line Options](https://github.com/mikf/gallery-dl/blob/master/docs/options.rst)
- [Instagram Extractor](https://github.com/mikf/gallery-dl/blob/master/docs/configuration.rst#extractorinstagram)

## ⚠️ Important Notes

1. **Authentication Required**: Most Instagram content requires authentication (cookies)
2. **Rate Limiting**: The node automatically applies rate limiting (1 second between requests)
3. **Dynamic Config**: The node creates temporary config files for Instagram options
4. **File Organization**: Files are automatically sorted into `images/`, `videos/`, `audio/`, `other/` folders
5. **Extra Options**: Use with caution - invalid options will cause downloads to fail

## 🧪 Testing

You can test the advanced options using the test script:
```bash
python Docs/test_advanced_options.py
```

This will verify that:
- Instagram include options work correctly
- Extra options are parsed properly
- Dynamic config files are created
- Command building works as expected

## 🎉 Summary

The Gallery-dl ComfyUI node now supports all major gallery-dl features:
- ✅ Instagram-specific content selection
- ✅ Custom command-line options
- ✅ Dynamic configuration
- ✅ Comprehensive debugging
- ✅ File organization
- ✅ Authentication support

You can now download specific types of Instagram content and use any gallery-dl command-line option through the ComfyUI interface!
