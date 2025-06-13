# ml-dist-gen
A Python script to generate token-based data distribution mixes for training multilingual language models. This script creates a training dataset by calculating sampling proportions based on the size of available data for each language, with options for fine-grained control.

## Input Data Format

The script requires an input file in `JSONL` format (e.g., `data.jsonl`), where each line is a separate JSON object describing a dataset. Each JSON object **must** contain the following keys:

| Key           | Type   | Description                                           |
| :------------ | :----- | :---------------------------------------------------- |
| `lang`        | String | The language code (e.g., `eng`, `deu`, `fra`).        |
| `dataset`     | String | The name of the dataset.                              |
| `gemma-3-tok` | Int    | The total number of tokens in this dataset.           |
| `path`        | String | The file path to the dataset.                         |

**Example `data.jsonl`:**
```json
{"lang": "eng", "dataset": "english_corpus_1", "gemma-3-tok": 125000000, "path": "data_tok/eng_1.jsonl"}
{"lang": "eng", "dataset": "english_corpus_2", "gemma-3-tok": 25000000, "path": "data_tok/eng_2.jsonl"}
{"lang": "deu", "dataset": "german_web_data", "gemma-3-tok": 98000000, "path": "data_tok/deu.jsonl"}
{"lang": "code", "dataset": "stack-edu", "gemma-3-tok": 50000000, "path": "data_tok/stack.jsonl"}
```

## Configuration
The core settings are hardcoded at the top of the run.py script. You can modify them directly to change the behavior of the distribution calculation.

```
# --- Hardcoded configurations ---
TOTAL_TRAINING_TOKENS = 4_000_000_000_000
DROP_CONFIG = {"eng": ["HPLT/HPLT2.0_cleaned"]}
MERGE_CONFIG = {"code": ["stack-edu", "starcoder"]}
FIXED_CONFIG = {"eng": 0.45, "code": 0.04, "math": 0.01}
MIN_THRESHOLD = 0.0005
```

- `TOTAL_TRAINING_TOKENS`: The total token budget for your planned training run. This is used to calculate the data usage summary.
- `DROP_CONFIG`: A dictionary to exclude specific datasets. The key is the language code (`lang`) and the value is a list of dataset names (`dataset`) to ignore.
- `MERGE_CONFIG`: A dictionary to group multiple datasets under a new, single language category. This is useful for creating logical groups like code.
- `FIXED_CONFIG`: A dictionary to assign a fixed, predefined proportion for specific languages. These languages are excluded from the dynamic calculation.
- `MIN_THRESHOLD`: The minimum proportion any language will receive in the final distribution. This ensures small datasets are not completely washed out.


## Usage

### 1. Calculate Basic Distribution

This is the default mode. It prints a table of languages and their calculated sampling proportions.

### Command:
```bash
python run.py data.jsonl
````

### Output:
```
Final Training Distribution

0.4499   eng
0.0781   spa
0.0745   deu
0.0629   fra
...
```

### 2. View the Data Usage Summary

Using the `--summary` flag, you can get a detailed report that includes the number of epochs each language will be trained on. This is critical for identifying languages that might be over-sampled.

### Command:
```bash 
python run.py data.jsonl --summary
```

### Output:
```
Final Training Distribution

0.4499   eng
0.0781   spa
0.0745   deu
0.0629   fra
...
========================================

Summary

Total Available Tokens: 5,485,459,417,085
Total Training Tokens: 8,000,000,000,000

High Usage Warning

One or more languages will be repeated more than 5 times.

Data Usage (Epochs per Language)

- !!! mlt: ~7.00 epochs
- !!! gle: ~6.21 epochs
-    eus: ~1.96 epochs
-    eng: ~1.80 epochs
```

### 3. Generate Training Data Paths

Using the `--path` flag, the script will generate a single-line string containing the calculated proportion followed by the dataset path, ready to be used as an argument in a training script.

### Command:
```bash
python run.py data.jsonl --path
```

### Output:

```
0.0745 data_tok/deu 0.0629 data_tok/fra 0.0781 data_tok/spa 0.0158 data_tok/ces 0.0068 data_tok/dan 0.0192 data_tok/ell 0.0089 data_tok/fin 0.0131 data_tok/hun ...
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
