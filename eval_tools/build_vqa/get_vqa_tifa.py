import glob
import json
import argparse
import torch
import transformers
import tqdm
import os

# formating prompt following LLaMA 2 style
def create_qg_prompt(caption):
    INTRO_BLURB = "Given an image description, generate one or two multiple-choice questions that verifies if the image description is correct.\nClassify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.\n"
    formated_prompt = f"<s>[INST] <<SYS>>\n{INTRO_BLURB}\n<</SYS>>\n\n"
    formated_prompt += f"Description: {caption} [/INST] Entities:"
    return formated_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--dataset", type=str, choices=["coco", "sgp_compbench", "vgbench"], required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)

args = parser.parse_args()

model_name = "tifa-benchmark/llama2_tifa_question_generation"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

with open(args.input_file, "r") as f:
    data = json.load(f)
    if args.dataset == "coco":
        captions = [x['caption'] for i, x in enumerate(data) if i % args.world_size == args.rank]
        data_ids = captions
    elif args.dataset == "sgp_compbench" or args.dataset == "vgbench":
        captions = [x['description'] for i, x in enumerate(data) if i % args.world_size == args.rank]
        data_ids = [x['file_name'].split(".")[0] for i, x in enumerate(data) if i % args.world_size == args.rank]
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

output_data = dict()

todo_idx = [i for i, x in enumerate(data_ids) if x not in output_data or 'output' in output_data[x]]
captions = [x for i, x in enumerate(captions) if i in todo_idx]
data_ids = [x for i, x in enumerate(data_ids) if i in todo_idx]
if os.path.exists(args.output_file):
    with open(args.output_file, 'r') as f:
        output_data = json.load(f)


todo_idx = [i for i, x in enumerate(data_ids) if x not in output_data or 'output' in output_data[x]]
captions = [x for i, x in enumerate(captions) if i in todo_idx]
data_ids = [x for i, x in enumerate(data_ids) if i in todo_idx]

print('Processing {} captions'.format(len(captions)))

# Process captions in batches
for i in tqdm.tqdm(range(0, len(captions), args.batch_size), total=(len(captions) + args.batch_size - 1) // args.batch_size):
    batch_captions = captions[i:i+args.batch_size]
    batch_data_ids = data_ids[i:i+args.batch_size]
    batch_prompts = [create_qg_prompt(cap) for cap in batch_captions]

    try:
        # Run batch inference
        batch_sequences = pipeline(
            batch_prompts,
            do_sample=False,
            num_beams=5,
            num_return_sequences=1,
            max_length=512
        )

        # Process each result in the batch
        for j, (cap, data_id, prompt, sequences) in enumerate(zip(batch_captions, batch_data_ids, batch_prompts, batch_sequences)):
            try:
                output = sequences[0]['generated_text'][len(prompt):]
                output = output.split('\n\n')[0]
                output_lines = output.split('\n')
                k = 0
                desc = []
                questions = []
                while (k < len(output_lines)):
                    if k + 3 < len(output_lines) and output_lines[k].startswith('About') \
                        and output_lines[k+1].startswith('Q:') and output_lines[k+2].startswith('Choices:') and output_lines[k+3].startswith('A:'):
                        try:
                            about = output_lines[k].split('About ')[1].split(' (')[0]
                            problem_cls = output_lines[k].split('About ')[1].split(' (')[1].split(')')[0]
                            question = output_lines[k+1].split('Q: ')[1]
                            choices = output_lines[k+2].split('Choices: ')[1].split(', ')
                            answer = output_lines[k+3].split('A: ')[1]
                            questions.append({
                                'about': about,
                                'cls': problem_cls,
                                'question': question,
                                'choices': choices,
                                'answer': answer,
                            })
                            flag = (answer in choices)
                            if not flag:
                                for c in choices:
                                    if answer in c or c in answer:
                                        answer = c
                                        flag = True
                                        break
                                if not flag:
                                    print(f'Answer {answer} not in choices {choices}')
                            assert flag
                        except Exception as e:
                            print(e)
                            print(f'Failed to process question in batch: {cap}')
                            print(output_lines[k:k+4])
                        k += 4
                    else:
                        desc.append(output_lines[k])
                        k += 1
                desc = '\n'.join(desc)
                output_data[data_id] = dict(
                    caption=cap,
                    description=desc,
                    questions=questions,
                )
                # import pdb; pdb.set_trace()
            except Exception as e:
                try:
                    output_data[data_id] = dict(
                        caption=cap,
                        output=output,
                    )
                except:
                    pass
                print(e)
                print(f'Failed to process caption in batch: {cap}')
                print(output)
    except Exception as e:
        print(e)
        print(f'Failed to process batch starting at index {i}')

#### Expected output ###
#  rabbit, plane
# Actibuilites:
# Colors: blue, red
# Counting:
# Other attributes:
# About rabbit (animal):
# Q: is this a rabbit?
# Choices: yes, no
# A: yes
# About rabbit (animal):
# Q: what animal is in the picture?
# Choices: rabbit, dog, cat, fish
# A: rabbit
# About plane (object):
# Q: is this a plane?
# Choices: yes, no
# A: yes
# About plane (object):
# Q: what type of vehicle is this?
# Choices: plane, car, motorcycle, bus
# A: plane
# About blue (color):
# Q: is the rabbit blue?
# Choices: yes, no
# A: yes
# About blue (color):
# Q: what color is the rabbit?
# Choices: blue, red, yellow, green
# A: blue
# About red (color):
# Q: is the plane red?
# Choices: yes, no
# A: yes
# About red (color):
# Q: what color is the plane?
# Choices: red, blue, yellow, green
# A: red


with open(args.output_file, 'w') as f:
    json.dump(output_data, f, indent=4)