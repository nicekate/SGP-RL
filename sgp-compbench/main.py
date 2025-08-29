from generation import *
from evaluation import *
from analysis import *
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Unified script for SVG generation, evaluation, and analysis")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help="Model name for SVG generation")
    parser.add_argument('--judge_model', type=str, default="gpt-4o-mini", help="Judge model name for evaluation")
    parser.add_argument('--bench', type=str, default="prompts/sgp-compbench-prompts.json", help="Benchmark JSON file path")
    parser.add_argument('--input', type=str, default=None, help="Input file path for SVG generation results")
    parser.add_argument('--output', type=str, default=None, help="Output file path for evaluation results")
    parser.add_argument('--workers', type=int, default=100, help="Number of worker processes for evaluation")
    parser.add_argument('--batch', type=int, default=50, help="Batch size for evaluation")
    parser.add_argument('--timeout', type=int, default=20, help="Timeout (seconds) for each evaluation entry")
    parser.add_argument('--no_generate', action='store_true', help="Skip SVG generation")
    parser.add_argument('--no_eval', action='store_true', help="Skip evaluation")
    parser.add_argument('--no_analysis', action='store_true', help="Skip analysis")
    parser.add_argument('--analyze_numeracy', action='store_true', help="Analyze numeracy-related metrics")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    if args.input is None:
        args.input = f"svg_generation_results/{model_name}_all_result.json"

    # 1. Generate SVGs
    if not args.no_generate:
        generate_all_svg(args.model, args.bench)

    # 2. Evaluate SVGs
    if not args.no_eval:
        process_file(
            args.input,
            args.bench,
            args.judge_model,
            args.output,
            args.workers,
            args.batch,
            args.timeout
        )

    # 3. Find the latest evaluation result file and analyze
    if not args.no_analysis:
        pattern = f"evaluation_results/result_{model_name}_all_result_*.json"
        matching_files = glob.glob(pattern)
        if matching_files:
            matching_files.sort(key=os.path.getmtime, reverse=True)
            file_path = matching_files[0]
        else:
            file_path = f"evaluation_results/result_{model_name}_all_result.json"
        output_dir = "analysis_results"
        analyze_results(file_path, output_dir, args.analyze_numeracy)

if __name__ == "__main__":
    main()