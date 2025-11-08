# Configuration Setup Guide

## Important: Protecting Your Personal Information

This repository includes **template configuration files** (`.example` files) that you should copy and customize with your own credentials. Your personal config files are automatically excluded from Git to keep your credentials safe.

## Quick Setup

### Step 1: Copy Template Files

Copy the `.example` files and remove the `.example` extension:

```powershell
# In the configs/ directory
cd configs

# Copy gallery-dl config template
Copy-Item gallery-dl.conf.example -Destination gallery-dl.conf

# Copy Instagram cookies template (if needed)
Copy-Item instagram_cookies.json.example -Destination instagram_cookies.json

# Copy auth config template (if needed)
Copy-Item auth_config.json.example -Destination auth_config.json
```

### Step 2: Add Your Credentials

Edit the copied files (without `.example`) and replace placeholder values with your actual credentials:

#### For `gallery-dl.conf`:
```json
{
  "extractor": {
    "reddit": {
      "client-id": "YOUR_REDDIT_CLIENT_ID",     // ← Replace with real values
      "client-secret": "YOUR_REDDIT_CLIENT_SECRET",
      "username": "YOUR_REDDIT_USERNAME",
      "password": "YOUR_REDDIT_PASSWORD"
    }
  }
}
```

#### For `instagram_cookies.json`:
Export cookies from your browser using a browser extension or developer tools.

### Step 3: Verify Protection

Your personal configs are safe because:

1. **They're in `.gitignore`** - Won't be committed to Git
2. **Template files are committed** - Users get examples without your data
3. **Clear documentation** - Users know how to set up their own

## What's Protected

These files will **NEVER** be committed to GitHub:

- `configs/gallery-dl.conf` (your personal file)
- `configs/gallery-dl-*.conf` (any variations)
- `configs/instagram_cookies.json`
- `configs/reddit_cookies.json`
- `configs/twitter_cookies.json`
- `configs/auth_config.json`
- Any `*.secret` or `*.key` files

## What's Committed to GitHub

These template files **WILL** be committed (they're safe):

- `configs/gallery-dl.conf.example` (template only)
- `configs/instagram_cookies.json.example` (template only)
- `configs/auth_config.json.example` (template only)
- `configs/yt-dlp.conf` (no credentials in yt-dlp configs)
- `configs/yt-dlp-*.conf` (safe config files)

## For New Users

When you clone this repository:

1. **Look for `.example` files** in the `configs/` directory
2. **Copy them** and remove `.example` from the filename
3. **Edit them** with your own credentials
4. **Never commit** the non-example versions

## Configuration Options

### Gallery-dl Authentication

For detailed authentication setup, see:
- `Docs/gallery_dl_authentication_guide.md`
- `Docs/gallery_dl_advanced_options_guide.md`

Supported methods:
- **Browser cookies** (automatic extraction)
- **Username/password** (direct credentials)
- **OAuth tokens** (for certain sites)
- **API keys** (for API-based access)

### yt-dlp Configuration

yt-dlp configs typically don't contain sensitive info, but can include:
- **Cookie extraction settings** (browser choice)
- **Output templates** (file naming)
- **Quality preferences**
- **Post-processing options**

See `Docs/yt_dlp_node_complete_guide.md` for details.

## Troubleshooting

### "Config file not found" error

Make sure you've copied the `.example` file:
```powershell
Copy-Item configs\gallery-dl.conf.example -Destination configs\gallery-dl.conf
```

### "Authentication failed" error

1. Check your credentials in the config file
2. Try browser cookie extraction instead: `"cookies-from-browser": "firefox"`
3. See authentication guide: `Docs/gallery_dl_authentication_guide.md`

### "Git wants to commit my credentials!"

This shouldn't happen if `.gitignore` is working. Check:
```powershell
# Test if file is ignored
git check-ignore configs/gallery-dl.conf

# Should output: configs/gallery-dl.conf
# If nothing appears, the file is NOT ignored (check .gitignore)
```

## Best Practices

1. **Never edit `.example` files directly** - They're templates for other users
2. **Keep credentials out of code** - Always use config files
3. **Use browser cookies when possible** - Easier and more secure than passwords
4. **Rotate credentials regularly** - Especially if shared on multiple machines
5. **Don't share screenshots** - They might show sensitive config data

## Getting Credentials

### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Choose "script" type
4. Copy client ID and secret

### Instagram
1. Use browser cookie extraction (easiest method)
2. Or use extension like "EditThisCookie" or "Cookie-Editor"
3. Export cookies to JSON format
4. See `Docs/gallery_dl_authentication_guide.md`

### Other Sites
See gallery-dl documentation: https://github.com/mikf/gallery-dl#authentication

## Security Notes

⚠️ **Important Security Warnings:**

1. **Don't commit credentials** - Even to private repos (they can leak)
2. **Use browser cookies** - They expire naturally, passwords don't
3. **Check .gitignore** - Before making your first commit
4. **Review Git history** - If you accidentally committed secrets, they stay in history
5. **Rotate compromised credentials** - If you accidentally shared them

## Need Help?

- **Authentication issues**: See `Docs/gallery_dl_authentication_guide.md`
- **Configuration options**: See `Docs/gallery_dl_advanced_options_guide.md`
- **yt-dlp setup**: See `Docs/yt_dlp_node_complete_guide.md`
- **General questions**: Open an issue on GitHub or contact eric@rollei.us

---

**Remember**: Template files (`.example`) are for sharing. Real config files are for you alone! 🔒
