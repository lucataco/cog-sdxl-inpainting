# diffusers/stable-diffusion-xl-1.0-inpainting-0.1 Cog model

This is an implementation of the [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@room.png -i mask=@room_mask.png -i prompt="modern bed with beige sheet and pillows"

## Example:

"modern bed with beige sheet and pillows"

![alt text](output.png)
