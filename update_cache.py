
#!/usr/bin/env python3

"""
Manual cache update script
Run this to update cache files when needed
"""

from cache_manager import update_all_caches

if __name__ == "__main__":
    print("🚀 Starting cache update...")
    update_all_caches()
    print("✅ Cache update completed!")
