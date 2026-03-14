import pandas as pd
from pathlib import Path
from utils.data_loader import DataLoader

# Your problematic example
test_data = """class,tweet_id,day,time,tweet,tweet_favorite_count,tweet_retweet_count,tweet_source,user_id,user_followers_count,user_friends_count,user_listed_count,user_statuses_count
SCHIZOPHRENIA,1535654073407512577,2022-06-11,16:03:56,\"""@USER I had a little dog BlancheðŸ"¼an American Pit bull came running at her, it was ready to kill her, I scooped her up &amp	 pushed her inside my jacket &amp	 faced this poor dog side on (its how he was trained - to kill) He was foaming at the mouth, he hurt me but no regrets. Just relief â¤ï¸ï¸ HTTPURL\""",5,0,Twitter for iPad,2959315666,466,845,4,4636"""

print("="*70)
print("TESTING PROBLEMATIC DATA PARSING")
print("="*70)

# Save to temp file
temp_file = Path('temp_test.csv')
with open(temp_file, 'w', encoding='utf-8') as f:
    f.write(test_data)

try:
    # Try standard pandas read
    print("\n📊 Attempting to parse with pandas...")
    df = pd.read_csv(temp_file, encoding='utf-8', quoting=1, on_bad_lines='skip')
    
    print(f"   ✓ Success! Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\n📝 First row:")
        for col in df.columns:
            value = df[col].iloc[0]
            if col == 'tweet':
                print(f"   {col}: {value[:100]}...")
            else:
                print(f"   {col}: {value}")
        
        # Test HTML decoding
        import html
        
        tweet_text = df['tweet'].iloc[0]
        decoded_text = html.unescape(tweet_text)
        
        print(f"\n🔧 After HTML decoding:")
        print(f"   Original: {tweet_text[:80]}...")
        print(f"   Decoded:  {decoded_text[:80]}...")
        
        # Test cleaning
        from utils.feature_extractor import clean_tweet_text
        
        cleaned = clean_tweet_text(decoded_text, remove_emojis=False, remove_mentions=True, remove_urls=True)
        
        print(f"\n🧹 After cleaning:")
        print(f"   {cleaned[:150]}...")
        
        print(f"\n✅ Problematic data can be processed!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up
    if temp_file.exists():
        temp_file.unlink()

print("\n" + "="*70)
print("Now testing with actual dataset...")
print("="*70)

# Test with actual data loader
loader = DataLoader()
users = loader.load_all_users()

if len(users) > 0:
    print(f"\n✅ Successfully loaded {len(users)} users from actual dataset!")
    
    # Show sample
    user = users[0]
    print(f"\n👤 Sample user:")
    print(f"   User ID: {user['user_id']}")
    print(f"   Class: {user['class_name']}")
    print(f"   Posts: {user['n_posts']}")
    print(f"\n   First post:")
    if len(user['posts']) > 0:
        print(f"   {user['posts']['text'].iloc[0][:150]}...")
else:
    print("\n⚠️  No users loaded from dataset")