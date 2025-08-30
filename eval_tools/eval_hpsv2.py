import json
import argparse
import glob
import os
import hpsv2
import tqdm
import numpy as np
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_file", type=str, default=None)

args = parser.parse_args()

with open(os.path.join(args.input_dir, 'id2img.json'), 'r') as f:
    id2img = json.load(f)
with open(os.path.join(args.input_dir, 'id2caption.json'), 'r') as f:
    id2caption = json.load(f)

if args.output_file is None:
    args.output_file = os.path.join(args.input_dir, "hpsv2.json")

result_dict = dict()

all_scores_mean = []
for image_id, image_paths in tqdm.tqdm(id2img.items()):
    prompt = id2caption[image_id]
    image_paths = [os.path.join(args.input_dir, p) for p in image_paths]
    try:
        scores = hpsv2.score(image_paths, prompt, hps_version='v2.1')
        scores = [float(score) for score in scores]
    except Exception as e:
        print(f"Error scoring {prompt}: {e}")
        try:
            scores = [float(hpsv2.score([image_paths[i]], prompt, hps_version='v2.1')[0]) for i in range(0, len(image_paths))]
        except Exception as e:
            print(f"Failed to score {prompt}: {e}")
            scores = [0] * len(image_paths)
    result_dict[image_id] = dict(
        hpsv2_score=scores,
        images=image_paths,
        hpsv2_score_mean=np.mean(scores),
        hpsv2_score_std=np.std(scores)
    )
    all_scores_mean.append(np.mean(scores))

with open(args.output_file, 'wb') as f:
    # json.dump(dict(
    #     results=result_dict,
    #     mean_hpsv2_score=np.mean(all_scores_mean)
    # ), f, indent=4)
    pkl.dump(result_dict, f)

print(f"Mean HPSv2 score for {args.result_dir}: {np.mean(all_scores_mean)}")
output_file_txt = args.output_file.replace('.json', '.txt')
with open(output_file_txt, 'w') as f:
    f.write(f"Mean HPSv2 score for {args.result_dir}: {np.mean(all_scores_mean)}\n")