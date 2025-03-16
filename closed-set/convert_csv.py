import csv
import json
from pathlib import Path

# Read the CSV file
csv_file = Path(__file__).parent / "Closed-Set-100.csv"
output_file = Path(__file__).parent / "whisper_input_descriptions.json"

# Define categories
CATEGORIES = ["Science", "Health", "Physics_Tech", "Society_Education", "Environment"]

texts = []
category_counters = {cat: 1 for cat in CATEGORIES}

with open(csv_file, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    # Skip header row
    next(csv_reader)
    
    # Process each row
    for row in csv_reader:
        # Process each column (category) in the row
        for category, text in zip(CATEGORIES, row):
            if text.strip():  # Only process non-empty texts
                # Create ID with category name and counter
                category_id = f"{category.lower()}_{category_counters[category]:03d}"
                texts.append({
                    "id": category_id,
                    "content": text.strip(),
                    "category": category
                })
                category_counters[category] += 1

# Create the output JSON structure
output_data = {
    "texts": texts
}

# Write to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Converted {len(texts)} texts to {output_file}")
print("\nTexts per category:")
for category in CATEGORIES:
    count = category_counters[category] - 1
    print(f"{category}: {count} texts") 