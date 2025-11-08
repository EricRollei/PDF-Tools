#!/usr/bin/env python3
"""Guide for creating a completely new Reddit app"""

def show_reddit_app_creation_guide():
    """Show step-by-step guide for creating a new Reddit app"""
    
    print("🆕 CREATING A NEW REDDIT APP (Step-by-Step)")
    print("=" * 60)
    
    print("""
📋 STEP 1: Clean Slate
   1. Go to: https://www.reddit.com/prefs/apps
   2. Find your old "Media Tool" app
   3. Click "delete app" to remove it completely
   4. This ensures no conflicts with old settings

📋 STEP 2: Check Account Settings  
   1. Go to: https://www.reddit.com/settings/privacy
   2. If "Two-factor authentication" is ON, turn it OFF
   3. Go to: https://www.reddit.com/settings/messaging
   4. Make sure your email is verified (green checkmark)

📋 STEP 3: Create New App
   1. Go back to: https://www.reddit.com/prefs/apps
   2. Click "Create App" or "Create Another App"
   3. Fill out the form EXACTLY like this:
   
      Name: GalleryDLTool2025
      App type: ● script (select this radio button)
      Description: (leave blank or put "Personal media downloader")
      About URL: (leave blank)
      Redirect URI: http://localhost:8080

   4. Click "Create app"

📋 STEP 4: Get New Credentials
   After creating the app, you'll see:
   • Client ID: The short string directly under "GalleryDLTool2025"
   • Client Secret: The longer string next to "secret"

📋 STEP 5: Test Format
   Your new config should look like:
   
   "reddit": {
     "client-id": "YOUR_NEW_CLIENT_ID",
     "client-secret": "YOUR_NEW_CLIENT_SECRET",
     "user-agent": "GalleryDLTool2025 (by /u/EricRollei)",
     "username": "EricRollei", 
     "password": "YOUR_REDDIT_PASSWORD"
   }

⚠️  CRITICAL CHECKLIST:
   ✓ App type is "script" (not "web app")
   ✓ 2FA is disabled on your Reddit account
   ✓ Email is verified on your Reddit account  
   ✓ Username matches exactly (case sensitive)
   ✓ Password is your current Reddit login password

🎯 ALTERNATIVE: Keep Using Browser Cookies
   If creating a new app is too much hassle, the browser cookie approach
   we proved works perfectly:
   • config_path: (empty)
   • use_browser_cookies: True
   • browser_name: firefox
   
   This downloaded 19 files successfully!
""")

def test_account_readiness():
    """Help user verify their Reddit account is ready"""
    
    print("\n🔍 ACCOUNT READINESS CHECKLIST")
    print("=" * 60)
    
    print("""
Please verify these settings on your Reddit account:

1. Username: EricRollei
   ➜ Go to https://www.reddit.com/settings/profile
   ➜ Confirm your username is exactly "EricRollei"

2. Email Verification:
   ➜ Go to https://www.reddit.com/settings/messaging  
   ➜ Look for green checkmark next to your email

3. Two-Factor Authentication:
   ➜ Go to https://www.reddit.com/settings/privacy
   ➜ If "Two-factor authentication" shows "ON", click to turn it OFF
   ➜ Reddit apps don't work with 2FA enabled

4. Password:
   ➜ Make sure you remember your current Reddit password
   ➜ Try logging out and back in to confirm it works

Once all these are verified, create the new app and test again.
""")

if __name__ == "__main__":
    show_reddit_app_creation_guide()
    test_account_readiness()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION")
    print("=" * 60)
    print("""
BEST APPROACH RIGHT NOW:
   
Since browser cookies are already working perfectly (19 files downloaded),
I recommend sticking with that approach for Reddit:

✅ WORKING CONFIG:
   • config_path: (leave empty)
   • use_browser_cookies: True  
   • browser_name: firefox

✅ FOR INSTAGRAM:
   • cookie_file: ./configs/instagram_cookies.json
   • use_browser_cookies: False

This gives you the best of both worlds without app credential hassles!
""")
