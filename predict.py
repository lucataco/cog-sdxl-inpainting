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

MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"
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

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="nike shoes in billboard",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy",
        ),
        image: Path = Input(
            description="Input image to inpaint.",
            default=None,
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White-colored regions are inpainted and black-colored regions are preserved",
        ),
        num_outputs: int = Input(
            description="Number of images to output. > 2 might generate out-of-memory errors.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed. If you need tweaks to a generated image, reuse the same seed number from output logs.",
            default=0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=1, le=500, default=100
        ),
        guidance_scale: float = Input(
            description="A higher guidance scale value generate images closely to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.", ge=1, le=20, default=7.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed <=0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")


        image = Image.open(image).convert("RGB").resize((512, 512))
        extra_kwargs = {
            "mask_image": Image.open(mask).convert("RGB").resize(image.size),
            "image": image
        }
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
