from  torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
from .svg import (extract_svg, safe_svg_to_image)

def render_svg_to_image(svg_code):
    """
    Attempts to parse and recover from errors in SVG code,
    then renders the recovered SVG to a PNG image.
    Parameters:
        svg_code (str): The SVG content as a string.
    """
    svg_content = extract_svg(svg_code)
        
    if not svg_content:
        
        return None
    
    try:
        image = safe_svg_to_image(svg_content)
        
        
        if image is None:
            
            return None
        
        if image.mode == 'RGBA':
            white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white_bg.paste(image, mask=image.split()[3])
            image =  white_bg.convert('RGB')
        elif image.mode != 'RGB':
            image =  image.convert('RGB') 
        
        return image
        
    except Exception as e:
        
        return None

class PromptImageDataset(Dataset):
    """Dataset for processing prompts."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_key,
        output_key=None,
        apply_chat_template=False,
        get_reference=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.get_reference = get_reference
        self.prompt_max_length = strategy.args.prompt_max_length

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        if get_reference:
            assert output_key is not None

        self.raw_prompts = []
        self.processed_prompts = []
        self.references = []

        def preprocess_data(data, input_key="input", apply_chat_template=None) -> str:
            if apply_chat_template:
                prompt = apply_chat_template(
                    [{"content": data[input_key], "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = data["prompt"]
            return data[input_key], prompt, data[output_key]

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            # print(data)
            prompt, processed_prompt, reference = preprocess_data(
                data, input_key, apply_chat_template
            )
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)
                if self.get_reference:
                    self.references.append(reference)
                    

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if self.get_reference:
            return (
                self.processed_prompts[idx],
                self.raw_prompts[idx],
                image_transform(Image.open(self.references[idx]).convert('RGB')),
            )
        return self.processed_prompts[idx], self.raw_prompts[idx]


class PromptSVGDataset(Dataset):
    """Dataset for processing prompts."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_key,
        output_key=None,
        apply_chat_template=False,
        get_reference=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.get_reference = get_reference
        self.prompt_max_length = strategy.args.prompt_max_length

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        if get_reference:
            assert output_key is not None

        self.raw_prompts = []
        self.processed_prompts = []
        self.references = []

        def preprocess_data(data, input_key="input", apply_chat_template=None) -> str:
            if apply_chat_template:
                prompt = apply_chat_template(
                    [{"content": data[input_key], "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = data["prompt"]
            return data[input_key], prompt, data[output_key]
        
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            
            prompt, processed_prompt, reference = preprocess_data(
                data, input_key, apply_chat_template
            )
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)
                if self.get_reference:
                    self.references.append(reference)
                    

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if self.get_reference:
            return (
                self.processed_prompts[idx],
                self.raw_prompts[idx],
                image_transform(render_svg_to_image(self.references[idx])),
            )
        return self.processed_prompts[idx], self.raw_prompts[idx]



# class PromptMixDataset(Dataset):
#     """Dataset for processing prompts."""

#     def __init__(
#         self,
#         dataset_dict,
#         mixture_ratio_dict,
#         tokenizer,
#         strategy,
#         input_key_dict,
#         output_key_dict,
#         apply_chat_template=False,
#     ) -> None:
#         super().__init__()
#         assert dataset_dict.keys() == mixture_ratio_dict.keys() == input_key_dict.keys() == output_key_dict.keys()
#         self.strategy = strategy
#         self.tokenizer = tokenizer
#         self.get_reference = True
#         self.prompt_max_length = strategy.args.prompt_max_length

#         if apply_chat_template:
#             apply_chat_template = self.tokenizer.apply_chat_template
        

#         self.raw_prompts = {k:[] for k in dataset_dict.keys()}   
#         self.processed_prompts = {k:[] for k in dataset_dict.keys()}  
#         self.references_math = {k:[] for k in dataset_dict.keys()}  
#         def preprocess_data(data, input_key="input", output_key = "output") -> str:
#             if apply_chat_template:
#                 prompt = apply_chat_template(
#                     [{"content": data[input_key], "role": "user"}],
#                     tokenize=False,
#                     add_generation_prompt=True,
#                 )
#             else:
#                 prompt = data["prompt"]
#             return data[input_key], prompt, data[output_key]
#         for k in dataset_dict.keys():
#             for data in tqdm(dataset_dict[k], disable=not self.strategy.is_rank_0()):
#                 # print(data)
#                 prompt, processed_prompt, reference = preprocess_data(
#                     data, input_key_dict[k], apply_chat_template
#                 )
#                 if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
#                     self.processed_prompts[k].append(processed_prompt)
#                     self.raw_prompts[k].append(prompt)
#                     if self.get_reference:
#                         self.references[k].append(reference)
                    

#     def __len__(self):
#         return len(self.raw_prompts)

#     def __getitem__(self, idx):
#         if self.get_reference:
#             return (
#                 self.processed_prompts[idx],
#                 self.raw_prompts[idx],
#                 image_transform(Image.open(self.references[idx]).convert('RGB')),
#             )
#         return self.processed_prompts[idx], self.raw_prompts[idx]
