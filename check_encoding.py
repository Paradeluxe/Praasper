import sys
print("Python default encoding:", sys.getdefaultencoding())

# Try different encodings
file_path = 'C:\\Users\\User\\Desktop\\Praasper\\.venv\\Lib\\site-packages\\panphon\\data\\ipa_all.csv'

for encoding in ['utf-8', 'cp950', 'gbk', 'latin1']:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            print(f"{encoding} works! Read {len(content)} characters")
        break
    except Exception as e:
        print(f"{encoding} failed:", e)

# Also try detecting with chardet
try:
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    print("chardet detection:", result)
except ImportError:
    print("chardet not available")