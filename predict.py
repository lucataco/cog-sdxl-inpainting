# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import math
import time
import torch
import subprocess
from PIL import Image
from typing import List
from diffusers import (AutoPipelineForInpainting,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler
)

MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
MODEL_CACHE = "checkpoints"
MODELS_URL = "https://weights.replicate.delivery/default/diffusers/sdxl-inpainting-0.1.tar"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Downloading weights")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODELS_URL, MODEL_CACHE)
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE,
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
        mask: Path = Input(description="Mask image - make sure it's the same size as the input image"),
        prompt: str = Input(
            description="Input prompt",
            default="modern bed with beige sheet and pillows",
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
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=4,
            default=1,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        input_image = self.scale_down_image(image, 1024)
        pil_mask = Image.open(mask)
        # Assume mask is same size as input image
        mask_image = pil_mask.resize((input_image.width, input_image.height))

        result = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
            width=input_image.width,
            height=input_image.height
        )

        output_paths = []
        for i, output in enumerate(result.images):
            output_path = f"/tmp/out-{i}.png"
            output.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
