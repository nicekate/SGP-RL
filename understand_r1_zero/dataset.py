from  torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])


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
