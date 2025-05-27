import os
import json
import argparse

def combine_json_files(input_dir, output_file):
    combined_json = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    key = os.path.splitext(filename)[0]
                    combined_json[key] = data
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")

    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(combined_json, out_file, indent=2)

    print(f"Combined JSON saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine JSON files from a directory into one JSON file.")
    parser.add_argument("--input_dir", type=str, help="Directory containing JSON files")
    parser.add_argument("--output_dir", type=str, help="Path to save the combined JSON file (including filename)")

    args = parser.parse_args()
    combine_json_files(args.input_dir, args.output_dir)
