# Gallery-dl Authentication Integration Guide

## Overview

Your Gallery-dl ComfyUI node now supports integrating with your existing authentication configuration file! This allows you to use stored credentials, cookies, and API keys from your web scraper setup.

## How Gallery-dl Cookie Extraction Works

### Browser Cookie Extraction
When you enable `use_browser_cookies`, gallery-dl uses the `browser-cookie3` Python library to:

1. **Locate Browser Profile**: Finds your browser's profile directory
2. **Extract Cookies**: Reads cookies from the browser's database files:
   - **Firefox**: `cookies.sqlite` 
   - **Chrome**: `Cookies` (SQLite database)
   - **Edge**: Similar Chrome-based structure
   - **Safari**: `Cookies.binarycookies`

3. **Filter by Domain**: Only sends cookies relevant to the target website
4. **HTTP Headers**: Adds cookies to requests as `Cookie: name=value` headers

### Supported Browsers
- Firefox (default)
- Chrome/Chromium  
- Microsoft Edge
- Safari (macOS)
- Opera

## Your Auth Config Integration

### How It Works
The node now accepts your `auth_config.json` file and automatically converts it to gallery-dl's configuration format:

```python
# Your auth_config.json format:
{
  "sites": {
    "reddit.com": {
      "auth_type": "api_client",
      "client_id": "your_client_id",
      "client_secret": "your_secret"
    }
  }
}

# Converted to gallery-dl format:
{
  "extractor": {
    "reddit": {
      "client-id": "your_client_id", 
      "client-secret": "your_secret"
    }
  }
}
```

### Supported Authentication Types

#### 1. Cookie Authentication
**Your Format:**
```json
"500px.com": {
  "auth_type": "cookie",
  "cookies": [
    {
      "name": "session_token",
      "value": "abc123...",
      "domain": ".500px.com"
    }
  ]
}
```

**Converted to Gallery-dl:**
```json
"500px": {
  "cookies": {
    "session_token": "abc123..."
  }
}
```

#### 2. Username/Password Authentication  
**Your Format:**
```json
"deviantart.com": {
  "auth_type": "login",
  "username": "your_username", 
  "password": "your_password"
}
```

**Converted to Gallery-dl:**
```json
"deviantart": {
  "username": "your_username",
  "password": "your_password"
}
```

#### 3. API Client (OAuth)
**Your Format:**
```json
"tumblr.com": {
  "auth_type": "oauth2",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "access_token": "your_token"
}
```

**Converted to Gallery-dl:**
```json
"tumblr": {
  "client-id": "your_client_id",
  "client-secret": "your_client_secret", 
  "access-token": "your_token"
}
```

#### 4. API Token
**Your Format:**
```json
"artstation.com": {
  "auth_type": "api_token",
  "token_value": "your_api_key"
}
```

**Converted to Gallery-dl:**
```json
"artstation": {
  "api-key": "your_api_key"
}
```

### Domain Mapping
The node automatically maps domains to gallery-dl extractor names:

| Your Domain | Gallery-dl Extractor |
|-------------|---------------------|
| reddit.com | reddit |
| imgur.com | imgur |
| deviantart.com | deviantart |
| artstation.com | artstation |
| twitter.com / x.com | twitter |
| instagram.com | instagram |
| flickr.com | flickr |
| pinterest.com | pinterest |
| 500px.com | 500px |
| pixiv.net | pixiv |
| behance.net | behance |
| bsky.app | bluesky |

## Usage Examples

### 1. Using Your Auth Config
```python
# In ComfyUI node:
auth_config_path = "a:\\Comfy_Dec\\ComfyUI\\custom_nodes\\Metadata_system\\configs\\auth_config.json"
url_list = "https://www.reddit.com/r/EarthPorn/top/"
```

### 2. Using Browser Cookies
```python
# Enable browser cookies for sites you're logged into
use_browser_cookies = True
browser_name = "firefox"  # or "chrome", "edge", etc.
```

### 3. Using Both (Recommended)
```python
# Auth config for API keys, browser cookies for session auth
auth_config_path = "path/to/auth_config.json"
use_browser_cookies = True
```

## Configuration Priority

1. **Auth Config**: Converted and applied first
2. **Explicit Config**: Your `config_path` overrides auth config 
3. **Browser Cookies**: Added regardless, works alongside other auth

## Benefits of Integration

### ✅ **Unified Authentication**
- Reuse existing credentials from your web scraper
- No need to duplicate auth setup
- Consistent authentication across tools

### ✅ **Multi-Method Support**
- API keys for official APIs
- Cookies for session-based auth
- Browser cookies for manual logins
- Username/password for basic auth

### ✅ **Automatic Conversion**
- Your auth config automatically converted
- No manual gallery-dl config needed
- Supports all your existing sites

### ✅ **Fallback Support**
- If auth config fails, browser cookies still work
- Multiple authentication methods can be combined
- Graceful degradation if credentials are invalid

## Example Usage in ComfyUI

1. **Set Auth Config Path**: 
   ```
   a:\Comfy_Dec\ComfyUI\custom_nodes\Metadata_system\configs\auth_config.json
   ```

2. **Add URLs**:
   ```
   https://www.reddit.com/r/EarthPorn/
   https://500px.com/popular
   https://www.deviantart.com/popular
   ```

3. **Enable Browser Cookies** (optional but recommended)

4. **Run**: The node will automatically use appropriate auth for each site!

## Troubleshooting

### Auth Config Issues
- ✅ Check file path is absolute
- ✅ Verify JSON syntax is valid
- ✅ Ensure domain names match exactly
- ✅ Check credentials are still valid

### Browser Cookie Issues  
- ✅ Make sure you're logged into the browser
- ✅ Try different browser selection
- ✅ Check browser is fully closed/reopened
- ✅ Verify browser profile permissions

### Mixed Authentication
- ✅ Auth config takes precedence over browser cookies
- ✅ Some sites work better with cookies vs API keys
- ✅ Try both methods if one fails

## Advanced Tips

### Site-Specific Notes

**Reddit**: Use API credentials from your auth config for higher rate limits

**500px**: Your existing cookies work perfectly  

**DeviantArt**: Browser cookies often work better than API

**Instagram**: Use browser cookies (API very restricted)

**Twitter/X**: Browser cookies recommended due to API changes

### Performance Optimization
- Use download archives to avoid re-downloading
- Set appropriate sleep delays (current: 1 second)
- Enable retry logic (current: 3 retries)

This integration gives you the best of both worlds - your existing authentication setup with gallery-dl's powerful downloading capabilities!
