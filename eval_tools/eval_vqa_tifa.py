from misc.tifa_score import tifa_score_benchmark
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="mplug-large")
parser.add_argument("--question_answer_path", type=str, default="/home/share/data/coco/coco_qa_tifa.json")
parser.add_argument("--result_dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    id2img_path = os.path.join(args.result_dir, "id2img.json")
    results = tifa_score_benchmark(args.model_name, args.question_answer_path, id2img_path, args.result_dir)
    with open(os.path.join(args.result_dir, f"tifa_scores_{args.model_name}.json"), "w") as f:
        json.dump(results, f, indent=4)