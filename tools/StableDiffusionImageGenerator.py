import diffusers
import transformers
import time
import torch
import os


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

        model_path = "SDModels/meinamix_meinaV11"

        if clip_skip > 1:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype = torch_dtype,
                text_encoder = self.text_encoder,
                safety_checker = None
            )
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype = torch_dtype,
                safety_checker=None
            )

        self.pipe = self.pipe.to(self.device_name)

        # Change the pipe scheduler to EADS.
        self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Prompt embeddings to overcome CLIP 77 token limit.
        # https://github.com/huggingface/diffusers/issues/2136

    def pad_sequence(self, sequence, batch_first=True, padding_value=0, max_length=None, device=None):
        if device is None:
            device = self.device_name

        if max_length is None:
            max_length = max([len(seq) for seq in sequence])

        padded_sequence = torch.full(
            (len(sequence), max_length), fill_value=padding_value, dtype=torch.long, device=device
        )
        for i, seq in enumerate(sequence):
            padded_sequence[i, : len(seq)] = seq.to(device)

        if batch_first:
            return padded_sequence
        else:
            return padded_sequence.transpose(0, 1)

    def get_prompt_embeddings(
            self,
            pipe,
            prompt,
            negative_prompt,
            device=None
    ):
        if device is None:
            device = self.device_name

        max_length = pipe.tokenizer.model_max_length

        input_ids = pipe.tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)

        shape_max_length = max(input_ids.shape[-1], negative_ids.shape[-1])

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=0, max_length=shape_max_length).to(device)
        negative_ids = self.pad_sequence(negative_ids, batch_first=True, padding_value=0, max_length=shape_max_length).to(
            device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(
                pipe.text_encoder(input_ids[:, i: i + max_length])[0].to(device)
            )
            neg_embeds.append(
                pipe.text_encoder(negative_ids[:, i: i + max_length])[0].to(device)
            )

        return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

    def generate_image(self, prompt, negative_prompt=None):

        if negative_prompt is None:
            negative_prompt = "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), extra arms, cleavage, sideboob"
        prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(
            self.pipe,
            prompt,
            negative_prompt,
            device = self.device_name
        )

        # Number of inference steps.
        num_inference_steps = 25

        # Guidance scale.
        guidance_scale = 10

        # Image dimensions - limited to GPU memory.
        width  = 512
        height = 512

        generator = torch.Generator(device=self.device_name)

        seed = generator.seed()

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

        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        img_path = f"image-{timestamp}.png"

        # if folder named "generated_images" doesn't exist, create it
        if not os.path.exists("generated_images"):
            os.makedirs("generated_images")
        img_path = os.path.join("generated_images", img_path)
        new_img[0].save(img_path)  # Save the image as a PNG file
        return img_path

# test this class
if __name__ == "__main__":
    style = "(intricate details), (****),"
    subject = "1girl, in 20s, cute, petite, short hair, bob cut, very detailed hair, blue hair, " \
              "detailed large eyes, sparkling eyes, glowing blue eyes, black sweater, long sleeves, black jeans,"
    action = "reading book"
    setting = "in a digital space"
    StableDiffusionImageGenerator().generate_image(style + ", " + subject + ", " + action + ", " + setting)
    StableDiffusionImageGenerator().generate_image(style + ", " + subject + ", " + action + ", " + setting)
