import diffusers
import transformers
import time
import torch

class StableDiffusionImageGenerator():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StableDiffusionImageGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):

        if torch.cuda.is_available():
            self.device_name = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            self.device_name = torch.device("cpu")
            torch_dtype = torch.float32


        clip_skip = 2

        if clip_skip > 1:
            self.text_encoder = transformers.CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder = "text_encoder",
                num_hidden_layers = 12 - (clip_skip - 1),
                torch_dtype = torch_dtype
            )

        # Load the pipeline.

        model_path = "SDModels/aingdiffusion_v90"

        if clip_skip > 1:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype = torch_dtype,
                safety_checker = None,
                text_encoder = self.text_encoder,
            )
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype = torch_dtype,
                safety_checker = None
            )

        self.pipe = self.pipe.to(self.device_name)

        # Change the pipe scheduler to EADS.
        self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Prompt embeddings to overcome CLIP 77 token limit.
        # https://github.com/huggingface/diffusers/issues/2136

    def get_prompt_embeddings(
        self,
        pipe,
        prompt,
        negative_prompt,
        split_character = ",",
        device = torch.device("cpu")
    ):
        max_length = pipe.tokenizer.model_max_length
        # Simple method of checking if the prompt is longer than the negative
        # prompt - split the input strings using `split_character`.
        count_prompt = len(prompt.split(split_character))
        count_negative_prompt = len(negative_prompt.split(split_character))

        # If prompt is longer than negative prompt.
        if count_prompt >= count_negative_prompt:
            input_ids = pipe.tokenizer(
                prompt, return_tensors = "pt", truncation = False
            ).input_ids.to(device)
            shape_max_length = input_ids.shape[-1]
            negative_ids = pipe.tokenizer(
                negative_prompt,
                truncation = False,
                padding = "max_length",
                max_length = shape_max_length,
                return_tensors = "pt"
            ).input_ids.to(device)

        # If negative prompt is longer than prompt.
        else:
            negative_ids = pipe.tokenizer(
                negative_prompt, return_tensors = "pt", truncation = False
            ).input_ids.to(device)
            shape_max_length = negative_ids.shape[-1]
            input_ids = pipe.tokenizer(
                prompt,
                return_tensors = "pt",
                truncation = False,
                padding = "max_length",
                max_length = shape_max_length
            ).input_ids.to(device)

        # Concatenate the individual prompt embeddings.
        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(
                pipe.text_encoder(input_ids[:, i: i + max_length])[0]
            )
            neg_embeds.append(
                pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
            )

        return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

    def generate_image(self, prompt, negative_prompt=None):

        if negative_prompt is None:
            negative_prompt = "(worst quality, low quality:1.4), monochrome, zombie, (interlocked fingers:1.2), extra arms,"
        prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(
            self.pipe,
            prompt,
            negative_prompt,
            split_character = ",",
            device = self.device_name
        )

        # Set to True to use prompt embeddings, and False to
        # use the prompt strings.
        use_prompt_embeddings = True

        # Seed and batch size.
        start_idx = 0
        batch_size = 1
        seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]

        # Number of inference steps.
        num_inference_steps = 20

        # Guidance scale.
        guidance_scale = 7

        # Image dimensions - limited to GPU memory.
        width  = 512
        height = 768

        images = []

        for count, seed in enumerate(seeds):
            start_time = time.time()

            if use_prompt_embeddings is False:
                new_img = self.pipe(
                    prompt = prompt,
                    negative_prompt = negative_prompt,
                    width = width,
                    height = height,
                    guidance_scale = guidance_scale,
                    num_inference_steps = num_inference_steps,
                    num_images_per_prompt = 1,
                    generator = torch.manual_seed(seed),
                ).images
            else:
                new_img = self.pipe(
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    width = width,
                    height = height,
                    guidance_scale = guidance_scale,
                    num_inference_steps = num_inference_steps,
                    num_images_per_prompt = 1,
                    generator = torch.manual_seed(seed),
                ).images


            images = images + new_img

        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        img_path = f"image-{timestamp}.png"

        images[0].save(img_path)  # Save the image as a PNG file
        return img_path

# test this class
if __name__ == "__main__":
    img_generator = StableDiffusionImageGenerator()
    img_generator.generate_image("""masterpiece, best quality, anime screencap, cute, petite,
short hair, bob cut, silver hair, blue eyes, denim, hoodie, halo, smile, long sleeves, hands on own face, 
wind, depth of field, forest, close-up,""")
    img_generator.generate_image("""masterpiece, best quality, anime screencap, cute, petite,
    short hair, bob cut, electric blue hair, blue eyes, cardigan sweater, smile, long sleeves, 
    depth of field, in massive library,""")
