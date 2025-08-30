import json
import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, choices=["coco", "sgp_compbench", "vgbench"], required=True)
parser.add_argument("--caption_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)

args = parser.parse_args()
if args.output_file is None:
    args.output_file = os.path.join(args.result_dir, "id2img.json")

with open(args.caption_file, 'r') as f:
    caption_data = json.load(f)

if args.dataset == "sgp_compbench":
    caption_data = {x['description']: x['file_name'].split(".")[0] for x in caption_data}
elif args.dataset == "coco":
    caption_data = caption_data['annotations']
    caption_data = {x['caption']: x['image_id'] for x in caption_data}
elif args.dataset == "vgbench":
    caption_data = {x['caption']: x['idx'] for x in caption_data}
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

batch_infos = glob.glob(f"{args.result_dir}/*/batch_info.json")

result_dict = dict()

for info_f in batch_infos:
    with open(info_f, 'r') as f:
        info = json.load(f)
    for res in info['prompts'].values():
        result_dict[caption_data[res['caption']]] = [resp['image_path'] for resp in res['responses'] if resp['valid']]

with open(args.output_file, 'w') as f:
    json.dump(result_dict, f, indent=4)