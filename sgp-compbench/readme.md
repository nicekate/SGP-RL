# SGP-CompBench

# Requirements

```bash
# install dependencies
pip install -r requirements.txt
```

# Replace with your own API key

```bash
# replace your own api key in util.py
base_url_generation = "your_base_url_generation"
key_generation = "your_key_generation"

base_url_eval = "your_base_url_eval" # gemini
key_eval = "your_key_eval"
```

# Usage

simple usage:
```bash
python main.py --model model_name
```

detailed usage:
```bash
python main.py --model model_name --judge_model judge_model --bench bench_file --input input_file --output output_file --workers workers --batch batch --timeout timeout
```

you can also skip some steps:
```bash
python main.py --model model_name --no_generate --no_eval --no_analysis
```