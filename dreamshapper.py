import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

model_id = "Lykon/dreamshaper-8"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_ghibli_style(input_image_path, strength=0.75, guidance_scale=7.5):
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))

    prompt = "ghibli style"
 
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]

    output_path = "ghibli_converted.png"
    result.save(output_path)
    print(f"Ghibli-style image saved to {output_path}")
    result.show()

if __name__ == "__main__":
   
    input_image = "img2.jpg"
    convert_to_ghibli_style(input_image)
