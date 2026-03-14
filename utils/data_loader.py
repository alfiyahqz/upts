import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from config import CLASS_MAPPING, CSV_COLUMNS, TIMESTAMP_FORMAT, DATASET_DIR

class DataLoader:
    """Load user CSV files and prepare for feature extraction"""
    
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.class_mapping = CLASS_MAPPING
        
    def load_all_users(self):
        """
        Load all user CSV files from dataset directory
        Handles HTML entities, encoding issues, and malformed CSVs
        
        Returns:
            users: List of dictionaries with user data
        """
        from utils.feature_extractor import clean_tweet_text
        from config import REMOVE_EMOJIS, REMOVE_MENTIONS, REMOVE_URLS
        import html
        
        users = []
        
        for class_name, label in self.class_mapping.items():
            class_dir = self.dataset_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_dir} not found, skipping {class_name}")
                continue
            
            csv_files = list(class_dir.glob("*.csv"))
            print(f"📁 Loading {len(csv_files)} files from {class_name}/")
            
            for csv_file in csv_files:
                try:
                    # Try different encoding and parsing strategies
                    df = None
                    
                    # Strategy 1: Standard CSV parsing with quote handling
                    try:
                        df = pd.read_csv(
                            csv_file, 
                            encoding='utf-8',
                            quoting=1,  # QUOTE_ALL
                            on_bad_lines='skip'  # Skip problematic lines
                        )
                    except Exception as e1:
                        # Strategy 2: Try with different encoding
                        try:
                            df = pd.read_csv(
                                csv_file, 
                                encoding='latin-1',
                                quoting=1,
                                on_bad_lines='skip'
                            )
                        except Exception as e2:
                            # Strategy 3: Read as text and manually parse
                            try:
                                df = self._parse_csv_manually(csv_file)
                            except Exception as e3:
                                print(f"❌ {csv_file.name}: Could not parse - {str(e3)}")
                                continue
                    
                    if df is None or len(df) == 0:
                        print(f"⚠️  {csv_file.name}: Empty or unparseable file")
                        continue
                    
                    # Validate columns
                    expected_cols = [
                        CSV_COLUMNS['text'],
                        CSV_COLUMNS['day'],
                        CSV_COLUMNS['time'],
                        CSV_COLUMNS['user_id']
                    ]
                    
                    # Check if we have the minimum required columns
                    missing_cols = [col for col in expected_cols if col not in df.columns]
                    
                    if missing_cols:
                        # Try to find columns by index if names don't match
                        if len(df.columns) >= 13:  # Expected number of columns
                            # Map by position
                            df.columns = [
                                'class', 'tweet_id', 'day', 'time', 'tweet',
                                'tweet_favorite_count', 'tweet_retweet_count',
                                'tweet_source', 'user_id', 'user_followers_count',
                                'user_friends_count', 'user_listed_count', 'user_statuses_count'
                            ][:len(df.columns)]
                        else:
                            print(f"⚠️  {csv_file.name}: Missing columns {missing_cols}")
                            continue
                    
                    # Get class from data if available
                    actual_class = class_name
                    if CSV_COLUMNS['class'] in df.columns:
                        csv_class = df[CSV_COLUMNS['class']].iloc[0]
                        if pd.notna(csv_class):
                            actual_class = str(csv_class).strip()
                            
                            # Check if class is in mapping
                            if actual_class not in self.class_mapping:
                                print(f"⚠️  {csv_file.name}: Unknown class '{actual_class}'")
                                # Try case-insensitive match
                                for key in self.class_mapping.keys():
                                    if key.lower() == actual_class.lower():
                                        actual_class = key
                                        break
                                else:
                                    print(f"   Skipping file with unknown class")
                                    continue
                            
                            label = self.class_mapping[actual_class]
                    
                    # Extract tweet text and decode HTML entities
                    if CSV_COLUMNS['text'] not in df.columns:
                        print(f"⚠️  {csv_file.name}: No 'tweet' column found")
                        continue
                    
                    df['tweet_decoded'] = df[CSV_COLUMNS['text']].apply(
                        lambda x: self._decode_html_entities(x) if pd.notna(x) else ''
                    )
                    
                    # Combine day and time
                    if CSV_COLUMNS['day'] in df.columns and CSV_COLUMNS['time'] in df.columns:
                        df['timestamp'] = df[CSV_COLUMNS['day']].astype(str) + ' ' + df[CSV_COLUMNS['time']].astype(str)
                    else:
                        print(f"⚠️  {csv_file.name}: Missing day or time column")
                        continue
                    
                    # Parse timestamps
                    df['timestamp'] = pd.to_datetime(
                        df['timestamp'], 
                        format=TIMESTAMP_FORMAT, 
                        errors='coerce'
                    )
                    
                    # Remove rows with invalid timestamps
                    df = df.dropna(subset=['timestamp'])
                    
                    if len(df) == 0:
                        print(f"⚠️  {csv_file.name}: No valid timestamps")
                        continue
                    
                    # Clean tweet text
                    df['text_cleaned'] = df['tweet_decoded'].apply(
                        lambda x: clean_tweet_text(
                            x, 
                            remove_emojis=REMOVE_EMOJIS,
                            remove_mentions=REMOVE_MENTIONS,
                            remove_urls=REMOVE_URLS
                        )
                    )
                    
                    # Filter out empty texts
                    df = df[df['text_cleaned'].str.strip() != '']
                    
                    if len(df) == 0:
                        print(f"⚠️  {csv_file.name}: No valid text after cleaning")
                        continue
                    
                    # Sort by timestamp
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Get user_id
                    user_id = 'unknown'
                    if CSV_COLUMNS['user_id'] in df.columns:
                        user_id_val = df[CSV_COLUMNS['user_id']].iloc[0]
                        if pd.notna(user_id_val):
                            user_id = str(user_id_val)
                    
                    if user_id == 'unknown':
                        user_id = csv_file.stem
                    
                    # Prepare final dataframe
                    posts_df = df[['text_cleaned', 'timestamp']].copy()
                    posts_df.columns = ['text', 'timestamp']
                    
                    users.append({
                        'user_id': user_id,
                        'label': label,
                        'class_name': actual_class,
                        'posts': posts_df,
                        'n_posts': len(posts_df),
                        'filename': csv_file.name
                    })
                    
                except Exception as e:
                    print(f"❌ Error loading {csv_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n✅ Loaded {len(users)} users total")
        
        if len(users) == 0:
            print("\n⚠️  No users loaded! Please check:")
            print("   1. Dataset directory path in config.py")
            print("   2. CLASS_MAPPING matches your folder names")
            print("   3. CSV files are not corrupted")
            return users
        
        # Print class distribution
        class_counts = {}
        for user in users:
            class_name = user['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\n📊 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} users")
        
        # Print post statistics
        post_counts = [user['n_posts'] for user in users]
        if post_counts:
            print(f"\n📊 Post statistics:")
            print(f"   Total posts: {sum(post_counts):,}")
            print(f"   Average posts per user: {np.mean(post_counts):.1f}")
            print(f"   Median posts per user: {np.median(post_counts):.0f}")
            print(f"   Min posts: {min(post_counts)}")
            print(f"   Max posts: {max(post_counts)}")
        
        return users

    def _decode_html_entities(self, text):
        """
        Decode HTML entities in text
        
        Args:
            text: Text with potential HTML entities
        
        Returns:
            decoded_text: Text with decoded entities
        """
        import html
        
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Decode HTML entities
        # &amp; -> &
        # &lt; -> 
        # &gt; -> >
        # &quot; -> "
        # etc.
        text = html.unescape(text)
        
        # Handle common encoding issues
        replacements = {
            'â¤ï¸ï¸': '❤️',
            'â¤': '❤',
            'ðŸ"¼': '🐕',
            'ðŸ˜­': '😭',
            'ðŸ¥º': '🥺',
            'ðŸ˜¤': '😤',
            'ðŸ¥°': '🥰',
            'ðŸ\'ž': '💞',
            'ðŸ\'•': '💕',
            'HTTPURL': '',  # Remove URL placeholder
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'â€¦': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def _parse_csv_manually(self, csv_file):
        """
        Manually parse CSV when pandas fails
        Handles cases where &amp; causes column splitting
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            df: Parsed DataFrame
        """
        import csv
        
        rows = []
        
        with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
            # Skip header
            header = f.readline().strip().split(',')
            
            for line in f:
                # Use csv module to handle quoted fields properly
                reader = csv.reader([line], quoting=csv.QUOTE_ALL)
                for row in reader:
                    if len(row) >= 13:  # Minimum expected columns
                        rows.append(row[:13])  # Take first 13 columns
        
        if not rows:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=[
            'class', 'tweet_id', 'day', 'time', 'tweet',
            'tweet_favorite_count', 'tweet_retweet_count',
            'tweet_source', 'user_id', 'user_followers_count',
            'user_friends_count', 'user_listed_count', 'user_statuses_count'
        ])
        
        return df
    
    def filter_by_post_count(self, users, min_posts=100, max_posts=None):
        """
        Filter users by number of posts
        
        Args:
            users: List of user dictionaries
            min_posts: Minimum number of posts required
            max_posts: Maximum number of posts (None = no limit)
        
        Returns:
            filtered_users: List of user dictionaries meeting criteria
        """
        filtered = []
        
        for user in users:
            n_posts = len(user['posts'])
            
            if n_posts < min_posts:
                continue
            
            if max_posts is not None and n_posts > max_posts:
                # Truncate to max_posts (keep most recent)
                user['posts'] = user['posts'].tail(max_posts).reset_index(drop=True)
            
            filtered.append(user)
        
        print(f"\n🔍 Filtered: {len(filtered)}/{len(users)} users have {min_posts}+ posts")
        return filtered

    def load_presplit_data(self, partition_csv_path):
        """
        Load data using existing partition file
        
        Args:
            partition_csv_path: Path to CSV with columns [filename, user_id, class, partition]
        
        Returns:
            users: List of user dictionaries with partition information
        
        Expected CSV format:
            filename,user_id,class,partition
            user_dep1.csv,user_dep1,depression,train
            user_dep2.csv,user_dep2,depression,val
            user_anx1.csv,user_anx1,anxiety,test
        """
        print(f"📂 Loading pre-split partition file: {partition_csv_path}")
        
        # Load partition CSV
        partition_df = pd.read_csv(partition_csv_path)
        
        # Validate columns
        required_cols = ['filename', 'user_id', 'class', 'partition']
        missing_cols = [col for col in required_cols if col not in partition_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"   Loaded {len(partition_df)} entries")
        
        # Print partition distribution
        print(f"\n📊 Partition distribution:")
        partition_counts = partition_df['partition'].value_counts()
        for partition, count in partition_counts.items():
            print(f"   {partition}: {count} users")
        
        # Print class distribution per partition
        print(f"\n📊 Class distribution per partition:")
        for partition in partition_df['partition'].unique():
            partition_data = partition_df[partition_df['partition'] == partition]
            class_counts = partition_data['class'].value_counts()
            print(f"\n   {partition}:")
            for cls, count in class_counts.items():
                print(f"      {cls}: {count} users")
        
        # Load users
        users = []
        
        for idx, row in partition_df.iterrows():
            filename = row['filename']
            user_id = row['user_id']
            class_name = row['class']
            partition = row['partition']
            
            # Find the CSV file
            # Try in class subdirectory first
            csv_path = self.dataset_dir / class_name / filename
            
            # If not found, try directly in dataset_dir
            if not csv_path.exists():
                csv_path = self.dataset_dir / filename
            
            # If still not found, search recursively
            if not csv_path.exists():
                found_paths = list(self.dataset_dir.rglob(filename))
                if found_paths:
                    csv_path = found_paths[0]
                else:
                    print(f"⚠️  Warning: File not found: {filename}, skipping")
                    continue
            
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                
                # Validate columns
                if CSV_COLUMNS['text'] not in df.columns:
                    print(f"⚠️  {filename}: Missing '{CSV_COLUMNS['text']}' column")
                    continue
                if CSV_COLUMNS['timestamp'] not in df.columns:
                    print(f"⚠️  {filename}: Missing '{CSV_COLUMNS['timestamp']}' column")
                    continue
                
                # Rename columns to standard names
                df = df.rename(columns={
                    CSV_COLUMNS['text']: 'text',
                    CSV_COLUMNS['timestamp']: 'timestamp'
                })
                
                # Parse timestamps
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=TIMESTAMP_FORMAT, errors='coerce')
                
                # Remove rows with invalid timestamps
                df = df.dropna(subset=['timestamp'])
                
                # Sort by timestamp (chronological order)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Filter out empty texts
                df = df[df['text'].str.strip() != '']
                
                if len(df) == 0:
                    print(f"⚠️  {filename}: No valid posts after cleaning")
                    continue
                
                # Get label from CLASS_MAPPING
                if class_name not in self.class_mapping:
                    print(f"⚠️  {filename}: Unknown class '{class_name}', skipping")
                    continue
                
                label = self.class_mapping[class_name]
                
                users.append({
                    'user_id': user_id,
                    'label': label,
                    'class_name': class_name,
                    'partition': partition,  # Add partition info
                    'posts': df[['text', 'timestamp']],
                    'filename': filename
                })
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
        
        print(f"\n✅ Successfully loaded {len(users)} users")
        
        # Verify all users have partition info
        users_without_partition = [u for u in users if 'partition' not in u]
        if users_without_partition:
            print(f"⚠️  Warning: {len(users_without_partition)} users missing partition info")
        
        return users


def verify_splits():
    """Verify that splits were created correctly"""
    from config import PROCESSED_DIR, CLASS_MAPPING
    
    print("=" * 60)
    print("VERIFYING SPLITS")
    print("=" * 60)
    
    split_dir = PROCESSED_DIR / "splits"
    
    if not split_dir.exists():
        print(f"\n❌ Error: Split directory not found: {split_dir}")
        print("   Please run 2_create_splits.py first.")
        return
    
    try:
        # Load all splits
        train_data = np.load(split_dir / "train.npz", allow_pickle=True)
        val_data = np.load(split_dir / "val.npz", allow_pickle=True)
        test_data = np.load(split_dir / "test.npz", allow_pickle=True)
        
        X_train, y_train, ids_train = train_data['X'], train_data['y'], train_data['user_ids']
        X_val, y_val, ids_val = val_data['X'], val_data['y'], val_data['user_ids']
        X_test, y_test, ids_test = test_data['X'], test_data['y'], test_data['user_ids']
        
        print(f"\n✅ Successfully loaded all splits")
        print(f"\n📊 Shapes:")
        print(f"   Train X: {X_train.shape}, y: {y_train.shape}")
        print(f"   Val   X: {X_val.shape}, y: {y_val.shape}")
        print(f"   Test  X: {X_test.shape}, y: {y_test.shape}")
        
        # Verify no overlap
        train_set = set(ids_train)
        val_set = set(ids_val)
        test_set = set(ids_test)
        
        assert len(train_set & val_set) == 0, "❌ Train-Val overlap detected!"
        assert len(train_set & test_set) == 0, "❌ Train-Test overlap detected!"
        assert len(val_set & test_set) == 0, "❌ Val-Test overlap detected!"
        
        print(f"\n✅ No data leakage (all splits are disjoint)")
        
        # Show class distribution
        reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
        
        print(f"\n📊 Class distribution:")
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            print(f"\n   {split_name}:")
            unique, counts = np.unique(split_y, return_counts=True)
            for label, count in zip(unique, counts):
                class_name = reverse_mapping.get(label, f"Unknown_{label}")
                percentage = (count / len(split_y)) * 100
                print(f"      {class_name} (label={label}): {count:3d} users ({percentage:5.1f}%)")
        
        # Show sample from each split
        print(f"\n📝 Sample users from each split:")
        print(f"   Train: {', '.join(ids_train[:5])}")
        print(f"   Val:   {', '.join(ids_val[:5])}")
        print(f"   Test:  {', '.join(ids_test[:5])}")
        
        print(f"\n✅ Verification complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run 2_create_splits.py first.")
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    users = loader.load_all_users()
    users = loader.filter_by_post_count(users, min_posts=100)
    
    # Show example
    if len(users) > 0:
        print("\n📝 Example user:")
        example = users[0]
        print(f"   User ID: {example['user_id']}")
        print(f"   Label: {example['label']} ({example['class_name']})")
        print(f"   Number of posts: {len(example['posts'])}")
        print(f"\n   First 3 posts:")
        print(example['posts'].head(3))