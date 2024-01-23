Stable Diffusion
================

Stable Diffusion in MLX. The implementation was ported from Hugging Face's
[diffusers](https://huggingface.co/docs/diffusers/index) and we are fetching
and using the weights available on the Hugging Face Hub by Stability AI at
[stabilitiai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

![out](generated-mlx.png)

*Image generated using Stable Diffusion in MLX and the prompt 'A big red sign saying MLX in capital letters.'*

Installation
------------

The dependencies are minimal, namely:

- `safetensors` and `huggingface-hub` to load the checkpoints.
- `regex` for the tokenization
- `numpy` because safetensors needs to return some form of array
- `tqdm` and `PIL` for the `txt2image.py` script

You can install all of the above with the `requirements.txt` as follows:

    pip install -r requirements.txt

Usage
------

Although each component in this repository can be used by itself, the fastest
way to get started is by using the `StableDiffusion` class from the `stable_diffusion`
module.

```python
from stable_diffusion import StableDiffusion

# This will download all the weights from HF hub and load the models in
# memory
sd = StableDiffusion()

# This creates a python generator that returns the latent produced by the
# reverse diffusion process.
#
# Because MLX is lazily evaluated iterating over this generator doesn't
# actually perform the computation until mx.eval() is called.
latent_generator = sd.generate_latents("A photo of an astronaut riding a horse on Mars.")

# Here we are evaluating each diffusion step but we could also evaluate
# once at the end.
for x_t in latent_generator:
    mx.simplify(x_t) # remove possible redundant computation eg reuse
                     # scalars etc
    mx.eval(x_t)

# Now x_t is the last latent from the reverse process aka x_0. We can
# decode it into an image using the stable diffusion VAE.
im = sd.decode(x_t)
```

The above is almost line for line the implementation of the `txt2image.py`
script in the root of the repository. You can use the script as follows:

    python txt2image.py "A photo of an astronaut riding a horse on Mars." --n_images 4 --n_rows 2

Image 2 Image
-------------

There is also the option of generating images based on another image using the
example script `image2image.py`. To do that an image is first encoded using the
autoencoder to get its latent representation and then noise is added according
to the forward diffusion process and the `strength` parameter. A `strength` of
0.0 means no noise and a `strength` of 1.0 means starting from completely
random noise.

![image2image](im2im.png)
*Generations with varying strength using the original image and the prompt 'A lit fireplace'.*

The command to generate the above images is:

    python image2image.py --strength 0.5 original.png 'A lit fireplace'

*Note: `image2image.py` will automatically downsample your input image to guarantee that its dimensions are divisible by 64. If you want full control of this process, resize your image prior to using the script.*

Performance
-----------

The following table compares the performance of the UNet in stable diffusion.
We report throughput in images per second **processed by the UNet** for the
provided `txt2image.py` script and the `diffusers` library using the MPS
PyTorch backend.

At the time of writing this comparison convolutions are still some of the least
optimized operations in MLX. Despite that, MLX still achieves **~40% higher
throughput** than PyTorch with a batch size of 16 and ~15% higher when
comparing the optimal batch sizes.

Notably, PyTorch achieves almost ~50% higher throughput for the batch size of 1
which is unfortunate as that means that a single image can be computed faster.
However, when starting with the models not loaded in memory and PyTorch's MPS
graph kernels not cached, the compilation time more than accounts for this
speed difference.

| Batch size |   PyTorch   |     MLX     |
| ---------- | ----------- | ----------- |
| 1          |  6.25 im/s  |  4.17 im/s  |
| 2          |  7.14 im/s  |  5.88 im/s  |
| 4          |**7.69 im/s**|  7.14 im/s  |
| 6          |  7.22 im/s  |  8.00 im/s  |
| 8          |  6.89 im/s  |  8.42 im/s  |
| 12         |  6.62 im/s  |  8.51 im/s  |
| 16         |  6.32 im/s  |**8.79 im/s**|

The above experiments were made on an M2 Ultra with PyTorch version 2.1,
diffusers version 0.21.4 and transformers version 4.33.3. For the generation we
used classifier free guidance which means that the above batch sizes result
double the images processed by the UNet.

Note that the above table means that it takes about 90 seconds to fully
generate 16 images with MLX and 50 diffusion steps with classifier free
guidance and about 120 for PyTorch.

Extensions
----------

This WIP PR adds some functionality to the command-line by augmenting the parser as can be found from the ```images2images.py --help```:
```
options:
  -h, --help            show this help message and exit
  --strength STRENGTH   value in (0,1); larger means more variation in the output image
  --n_images N_IMAGES   total number of images arranged in n_rows
  --n_rows N_ROWS       the number of rows in the grid of final images
  --steps STEPS         maximum number of steps N
  --cfg CFG             configuration number N
  --negative_prompt NEGATIVE_PROMPT
                        things to avoid in the final images
  --decoding_batch_size DECODING_BATCH_SIZE
  -o OUTPUT, --output OUTPUT
                        base.ext filename for outputs
  -sd SHOW_DENOISING, --show_denoising SHOW_DENOISING
                        show denoising images every N iterations
  -sp SAVE_PROMPT, --save_prompt SAVE_PROMPT
                        save the main test-prompt as metadata
  --save_last_N SAVE_LAST_N
                        save all the last N consecutive sets of images
  -pp PRINT_PARSER, --print_parser PRINT_PARSER
                        print the argument Namespace at inception
  -gt GENERATE_TRIANGLE, --generate_triangle GENERATE_TRIANGLE
                        create images for a triangular display of progressive strength
```

Specifically it allows the use of `-pp` to display the args parser parameters at inception; `-sp` to save the text-prompt as metadata with the file; `-gt` to generate the images necessary for the triangular array format below that illustrates the diffusion process from start to finish; `-sd N` to set the interval over which the intermediate noisy images are saved to separate files, for example with `--steps 80` and `-sd 20` the images will be saved with the default `args.output` as `output_80_20.png`, `output_80_40.png` etc which will be picked up by the helper script `TriangularArray.py` (WIP - still needs some manual intervention until all the image-file parsing is defined) to create an image such as this with the seed image left and the final diffusion image right. Running along a row gives an impression of the way the noising and denoising evolve, although it appears from the `__init__.py`code that the `add_noise` is in practice done all at once; it's just done to a greater extent as the `--steps` and `--strength` values rise to take us deeper and deeper into the latent space (if I haver understood this correctly):

![triangular_diffusion_array](./images2images/tpyramid_pattern6_gh.png)

*Images generated using progressive values of `--strength` from `0.1` to `1.0` in increments of `0.1` using Stable Diffusion in MLX and the prompt 'Idyllic country landscape. Impressionism. Style of Cezanne.' The idea was to visualise the way the noising takes the seed image (left) deeper into latent space and the denoising the retrieves the target image (right). Rows represent separate `--steps` and `--strength` settings generating total steps of [20, 40, ... 200] in the ten rows.*

There are constraints on the image sizes that can be processed on an `M2 MAX 32GB` and although large rectangular images on low numbers of steps are possible, the ideal size and shape to do a large number of steps up to around `--steps 200` appears to be `[512, 512, 3]` although larger images do work sometimes.

`--save_last_N` is a utility to save the last consecutive `N` images before the final image for any value of `--strength`. Motivation: sometimes the final image is very smooth and may be thought less attractive; the option of seeing the final `N` images allows others to be chosen.