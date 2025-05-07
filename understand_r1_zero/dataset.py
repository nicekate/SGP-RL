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


class PureTextDataset(Dataset):
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
            return data[input_key], prompt

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            # print(data)
            prompt, processed_prompt = preprocess_data(
                data, input_key, apply_chat_template
            )
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)
                
                    

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if self.get_reference:
            return (
                self.processed_prompts[idx],
                self.raw_prompts[idx],
                image_transform(Image.new('RGB', (224, 224), color=(0, 0, 0))),
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

class PromptImageSVGDataset(Dataset):
    """
    Dataset for processing prompts that can handle both image paths (COCO) 
    and SVG code (HaoQuan SVG) based on the source.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_key,
        output_key=None,
        svg_key="svg",
        image_path_key="image_path",
        dataset_source_key="dataset_source",
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
        # if get_reference:
        #     assert output_key is not None

        self.raw_prompts = []
        self.processed_prompts = []
        self.references = []
        self.data_sources = []  # Track source (coco or svg)

        def preprocess_data(data, input_key="input", apply_chat_template=None) -> tuple:
            if apply_chat_template:
                prompt = apply_chat_template(
                    [{"content": data[input_key], "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = data["prompt"]
                
            # Determine data source
            data_source = data.get(dataset_source_key, None)
            
            # If source isn't explicitly labeled, infer from fields
            if data_source is None:
                if data.get(svg_key) is not None:
                    data_source = "svg"
                elif data.get(image_path_key) is not None:
                    data_source = "coco"
                else:
                    raise ValueError(f"Cannot determine data source for entry: {data}")
                
            # Get reference based on source
            if data_source == "svg":
                reference = data.get(svg_key)
            else:  # coco
                reference = data.get(image_path_key)
                
            return data[input_key], prompt, reference, data_source

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, processed_prompt, reference, data_source = preprocess_data(
                data, input_key, apply_chat_template
            )
            
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)
                if self.get_reference:
                    self.references.append(reference)
                    self.data_sources.append(data_source)
                    
        print(f"Dataset loaded with {len(self.raw_prompts)} entries "
              f"({sum(1 for s in self.data_sources if s == 'coco')} COCO, "
              f"{sum(1 for s in self.data_sources if s == 'svg')} SVG)")

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if not self.get_reference:
            return self.processed_prompts[idx], self.raw_prompts[idx]
            
        # Process based on data source
        if self.data_sources[idx] == "svg":
            # For SVG: Render SVG code to image
            try:
                image = render_svg_to_image(self.references[idx])
                if image is None:
                    # Fallback to an empty image if rendering fails
                    image = Image.new('RGB', (224, 224), color=(255, 255, 255))
                transformed_image = image_transform(image)
            except Exception as e:
                print(f"Error rendering SVG at index {idx}: {e}")
                transformed_image = image_transform(Image.new('RGB', (224, 224), color=(255, 255, 255)))
        else:
            # For COCO: Load image from disk
            try:
                image = Image.open(self.references[idx]).convert('RGB')
                transformed_image = image_transform(image)
            except Exception as e:
                print(f"Error loading image at {self.references[idx]}: {e}")
                transformed_image = image_transform(Image.new('RGB', (224, 224), color=(255, 255, 255)))
                
        return self.processed_prompts[idx], self.raw_prompts[idx], transformed_image

