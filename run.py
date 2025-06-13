import json
import argparse
from collections import defaultdict

# --- Hardcoded configurations ---
TOTAL_TRAINING_TOKENS = 4_000_000_000_000
DROP_CONFIG = {"eng": ["HPLT/HPLT2.0_cleaned"]}
MERGE_CONFIG = {"code": ["stack-edu", "starcoder"]}
FIXED_CONFIG = {"eng": 0.45, "code": 0.04, "math": 0.01}
MIN_THRESHOLD = 0.0005

def compute_distribution(
    jsonl_path,
    total_training_tokens,
    drop_datasets_per_lang=None,
    merge_datasets=None,
    fixed_proportions=None,
    min_threshold=None,
):
    """
    Computes the training distribution for a multilingual model.
    """
    drop_datasets_per_lang = drop_datasets_per_lang or {}
    merge_datasets = merge_datasets or {}
    fixed_proportions = fixed_proportions or {}

    # Data loading
    try:
        data = [json.loads(line) for line in open(jsonl_path)]
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{jsonl_path}' was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error: The file '{jsonl_path}' is not a valid JSONL file.")

    # Validating and datasets dropping 
    filtered_data = []
    for entry in data:
        lang, ds, path, tokens = entry.get("lang"), entry.get("dataset"), entry.get("path"), entry.get("gemma-3-tok")
        if not all([lang, ds, path, isinstance(tokens, int)]):
            raise ValueError(f"Invalid or missing key in entry (lang, dataset, path, gemma-3-tok required): {entry}")
        if ds not in drop_datasets_per_lang.get(lang, []):
            filtered_data.append(entry)

    # Merging datasets by language
    lang_to_tokens = defaultdict(int)
    for entry in filtered_data:
        lang_to_tokens[entry["lang"]] += entry["gemma-3-tok"]

    # Merge datasets across languages
    dataset_to_new_iso = {ds: new_lang for new_lang, datasets in merge_datasets.items() for ds in datasets}
    merged_lang_tokens = defaultdict(int)
    for entry in filtered_data:
        new_lang = dataset_to_new_iso.get(entry["dataset"], entry["lang"])
        merged_lang_tokens[new_lang] += entry["gemma-3-tok"]

    total_available_tokens = sum(merged_lang_tokens.values())

    fixed_sum = sum(fixed_proportions.values())
    if fixed_sum > 1:
        raise ValueError(f"Fixed proportions sum to more than 1: {fixed_sum}")

    # Distribution calculation
    leftover_tokens = sum(tokens for lang, tokens in merged_lang_tokens.items() if lang not in fixed_proportions)
    distribution = {}
    if leftover_tokens > 0:
        for lang, tokens in merged_lang_tokens.items():
            distribution[lang] = fixed_proportions.get(lang, (tokens / leftover_tokens) * (1 - fixed_sum))
    else:
        distribution = fixed_proportions.copy()

    # Minimum threshold applying
    if min_threshold:
        bumps = {lang: min_threshold for lang, val in distribution.items() if lang not in fixed_proportions and val < min_threshold}
        if bumps:
            bump_total = sum(bumps.values())
            to_adjust = {lang: val for lang, val in distribution.items() if lang not in fixed_proportions and lang not in bumps}
            adjust_total = sum(to_adjust.values())
            if adjust_total > 0:
                scale_factor = (1 - fixed_sum - bump_total) / adjust_total
                adjusted = {lang: val * scale_factor for lang, val in to_adjust.items()}
                distribution = {**fixed_proportions, **bumps, **adjusted}

    # Normalization and rounding of language-level distribution
    total_prop = sum(distribution.values())
    if total_prop > 0:
        distribution = {lang: val / total_prop for lang, val in distribution.items()}
    rounded = {k: round(v, 4) for k, v in distribution.items()}
    diff = round(1.0 - sum(rounded.values()), 4)
    if diff != 0 and rounded:
        key = max(rounded, key=rounded.get)
        rounded[key] = round(rounded[key] + diff, 4)

    # Report generation
    language_usage_report = {
        lang: (proportion * total_training_tokens) / merged_lang_tokens.get(lang, 1)
        for lang, proportion in rounded.items() if merged_lang_tokens.get(lang, 0) > 0
    }

    # High-precision proportions generation for each path
    dataset_proportions = {}
    for entry in filtered_data:
        final_lang = dataset_to_new_iso.get(entry["dataset"], entry["lang"])
        lang_group_proportion = rounded.get(final_lang)
        lang_group_total_tokens = merged_lang_tokens.get(final_lang)
        if lang_group_proportion is not None and lang_group_total_tokens is not None and lang_group_total_tokens > 0:
            dataset_proportions[entry["path"]] = lang_group_proportion * (entry["gemma-3-tok"] / lang_group_total_tokens)

    # Normalization and rounding of dataset-specific proportions
    final_dataset_proportions = {path: round(prop, 4) for path, prop in dataset_proportions.items()}
    diff = round(1.0 - sum(final_dataset_proportions.values()), 4)
    if diff != 0 and final_dataset_proportions:
        key_to_adjust = max(final_dataset_proportions, key=final_dataset_proportions.get)
        final_dataset_proportions[key_to_adjust] = round(final_dataset_proportions[key_to_adjust] + diff, 4)


    return {
        "distribution": rounded,
        "total_available_tokens": total_available_tokens,
        "language_usage_report": language_usage_report,
        "dataset_proportions": final_dataset_proportions, # Return rounded proportions
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute training distribution for a multilingual model.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("jsonl_path", help="Path to the JSONL file with token counts.")
    parser.add_argument("--summary", action="store_true", help="Print a detailed summary including token counts and usage warnings.")
    parser.add_argument("--path", action="store_true", help="Generate a single line output with proportion and path for each dataset.")
    args = parser.parse_args()

    try:
        result = compute_distribution(
            jsonl_path=args.jsonl_path,
            total_training_tokens=TOTAL_TRAINING_TOKENS,
            drop_datasets_per_lang=DROP_CONFIG,
            merge_datasets=MERGE_CONFIG,
            fixed_proportions=FIXED_CONFIG,
            min_threshold=MIN_THRESHOLD,
        )

        # --- Output Logic ---
        if args.path:
            # Use format specifier to ensure 4 decimal places are always printed
            output_parts = [f"{prop:.4f} {path}" for path, prop in result["dataset_proportions"].items()]
            print(" ".join(output_parts))
        else:
            print("Final Training Distribution\n")
            sorted_dist = sorted(result["distribution"].items(), key=lambda item: item[1], reverse=True)
            for lang, proportion in sorted_dist:
                print(f"{proportion:<8.4f} {lang}")
        
        if args.summary and not args.path:
            print("\n" + "="*40 + "\n")
            print("Summary\n")
            print(f"Total Available Tokens: {result['total_available_tokens']:,}")
            print(f"Total Training Tokens: {TOTAL_TRAINING_TOKENS:,}\n")

            usage_report = result['language_usage_report']
            overused_languages = {lang: usage for lang, usage in usage_report.items() if usage > 5}

            if overused_languages:
                print("High Usage Warning\n")
                print("One or more languages will be repeated more than 5 times.\n")
            else:
                print("Usage Check\n")
                print("All languages are within the desired usage limit (<= 5 times).\n")

            print("Data Usage (Epochs per Language)\n")
            sorted_usage = sorted(usage_report.items(), key=lambda item: item[1], reverse=True)
            for lang, usage in sorted_usage:
                marker = "!!!" if lang in overused_languages else "  "
                print(f"- {marker} {lang}: ~{usage:.2f} epochs")

    except (FileNotFoundError, ValueError) as e:
        print(e)