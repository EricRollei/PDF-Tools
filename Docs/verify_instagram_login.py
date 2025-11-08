"""
Quick Instagram login verification script
Run this after logging into Instagram to verify cookies are working
"""

def verify_instagram_login():
    """Verify Instagram login cookies are available"""
    print("🔐 Instagram Login Verification")
    print("="*40)
    
    try:
        import browser_cookie3
        
        # Check Firefox
        print("🦊 Checking Firefox cookies...")
        firefox_cookies = list(browser_cookie3.firefox())
        instagram_cookies = [c for c in firefox_cookies if 'instagram.com' in c.domain]
        
        if instagram_cookies:
            print(f"✅ Found {len(instagram_cookies)} Instagram cookies in Firefox")
            
            # Check for session cookie
            session_cookies = [c for c in instagram_cookies if c.name == 'sessionid']
            if session_cookies:
                print("✅ Session cookie found - you're logged in!")
                print("🎯 Ready for Gallery-dl downloads")
            else:
                print("⚠️ No session cookie - try logging in again")
        else:
            print("❌ No Instagram cookies found")
            print("💡 Please log into Instagram in Firefox")
            
        # Check Chrome (if running as admin)
        try:
            print("\n🔍 Checking Chrome cookies...")
            chrome_cookies = list(browser_cookie3.chrome())
            instagram_chrome = [c for c in chrome_cookies if 'instagram.com' in c.domain]
            
            if instagram_chrome:
                print(f"✅ Found {len(instagram_chrome)} Instagram cookies in Chrome")
                session_chrome = [c for c in instagram_chrome if c.name == 'sessionid']
                if session_chrome:
                    print("✅ Chrome session cookie found - you're logged in!")
            else:
                print("❌ No Instagram cookies in Chrome")
                
        except Exception as e:
            print(f"⚠️ Chrome cookie access failed: {e}")
            print("💡 Run ComfyUI as Administrator for Chrome cookie access")
            
    except ImportError:
        print("❌ browser_cookie3 not available")

if __name__ == "__main__":
    verify_instagram_login()
