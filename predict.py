# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import (AutoPipelineForInpainting,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler
)

MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_CACHE = "model-cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

    def scale_down_image(self, image_path, max_size):
        image = Image.open(image_path)
        width, height = image.size
        scaling_factor = min(max_size/width, max_size/height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2)
            )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))
    
    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="monochrome, lowres, bad anatomy, worst quality, low quality",
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=10, default=8.0
        ),
        steps: int = Input(
            description="Number of denoising steps", ge=1, le=80, default=20),
        strength: float = Input(
            description="1.0 corresponds to full destruction of information in image", ge=0.01, le=1.0, default=0.7),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        img = Image.open(image)
        input_image = self.scale_down_image(image, 1024)
        input_image.save("out-1.png")

        mas = Image.open(mask)
        # Assume mask is same size as input image
        mask_image = mas.resize((input_image.width, input_image.height))
        mask_image.save("out-2.png")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
            width=input_image.width,
            height=input_image.height
        ).images[0]

        output_path = "output.png"
        result.save(output_path)
        return Path(output_path)
