import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs="+", required=True)
parser.add_argument("--dataset", type=str, choices=["coco", "sgp_compbench", "vgbench"], required=True)
parser.add_argument("--caption_file", type=str, default=None, help="The caption file for the COCO dataset")
parser.add_argument("--output_file", type=str, required=True)

args = parser.parse_args()

question_data = dict()

for input_file in args.input_files:
    with open(input_file, "r") as f:
        question_data.update(json.load(f))

if args.dataset == "coco":
    with open(args.caption_file, 'r') as f:
        caption_data = json.load(f)

    qa_data = []

    caption_data = caption_data['annotations']
    caption_data = {x['caption']: x['image_id'] for x in caption_data}

    for cap, q in question_data.items():
        assert cap in caption_data, f"Caption {cap} not found in caption data"
        for q in q['questions']:
            qa_data.append({
                'id': caption_data[cap],
                'question': q['question'],
                'choices': q['choices'],
                'answer': q['answer'],
                'element_type': q['cls'],
            })
else:
    qa_data = []

    for idx, q in question_data.items():
        # assert cap in caption_data, f"Caption {cap} not found in caption data"
        for q in q['questions']:
            qa_data.append({
                'id': idx,
                'question': q['question'],
                'choices': q['choices'],
                'answer': q['answer'],
                'element_type': q['cls'],
            })

    with open(args.output_file, 'w') as f:
        json.dump(qa_data, f, indent=4)
