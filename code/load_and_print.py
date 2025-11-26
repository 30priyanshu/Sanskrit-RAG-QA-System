import os

DATA_DIR = "../data"

# List all .txt files in the data directory
for fn in os.listdir(DATA_DIR):
    if fn.endswith('.txt'):
        full_path = os.path.join(DATA_DIR, fn)
        print(f"--- {fn} ---")
        with open(full_path, encoding='utf-8') as file:
            content = file.read()
            print(content)
        print()
