# Chunkmogrify: Real image inversion via Segments

<div>
  <p align="center">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/chunkmogrify/logo.png" alt="Logo">
  </p>
</div>


<p align="center">
Teaser video with live editing sessions can be found here  <br/>
  <a href="https://youtu.be/watch?v=JU1OVGCl6so">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/chunkmogrify/teaser_thumbnail.jpg" />
  </a>
</p>



This code demonstrates the ideas discussed in arXiv submission *Real Image Inversion via Segments*.  
http://arxiv.org/abs/2110.06269  
(David Futschik, Michal Lukáč, Eli Shechtman, Daniel Sýkora)

*Abstract:  
We present a simple, yet effective approach to editing
real images via generative adversarial networks (GAN). Unlike previous
techniques, that treat all editing tasks as an operation that affects pixel
values in the entire image in our approach we cut up the image into a set of
smaller segments. For those segments corresponding latent codes of a generative
network can be estimated with greater accuracy due to the lower number of
constraints. When codes are altered by the user the content in the image is
manipulated locally while the rest of it remains unaffected. Thanks to this
property the final edited image better retains the original structures and thus
helps to preserve natural look.*

<div>
  <p align="center">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/00104.jpg" width="350" alt="before">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/00101.jpg" width="350" alt="after">
  </p>
</div>

<div>
  <p align="center">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/dfutschik.jpg" width="350" alt="before">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/00009.jpg" width="350" alt="after">
  </p>
</div>

## What do I need?
You will need a local machine with a relatively recent GPU - I wouldn't recommend trying
Chunkmogrify with anything older than RTX 2080. It is technically possible to run even on CPU,
but the operations become so slow that the user experience is not enjoyable.

## Quick startup guide
Requirements:  
Python 3.7 or newer
  
Note: If you are using Anaconda, I recommend creating a new environment to run this project.
Packages installed with conda and pip often don't play together very nicely.  
  
Steps to be able to successfully run the project:  
1) Clone or download the repository and open a terminal / Powershell instance in the directory.  
2) Install the required python packages by running `pip install -r requirements.txt`. This
might take a while, since it will download a few packages which will be several hundred MBs of data.
Some packages might need to compile their extensions (as well as this project itself), so a C++
compiler needs to be present. On Linux, this is typically not an issue, but running on Windows might
require Visual Studio and CUDA installations to successfully setup the project.
3) Run `python app.py`. When running for the first time, it will automatically download required
resources, which are also several hundred megabytes. Progression of the download can be monitored
in the command line window.  
  
To see if everything installed and configured properly, load up a photo and try running a projection
step. If there are no errors, you are good to go.  


### Possible problems:
*Torch not compiled with CUDA enabled.*  
Run
```
pip uninstall torch
pip cache purge
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
```

## Explanation of usage

<p align="center">
Tutorial video: click below  <br/>
  <a href="https://youtube.com/watch?v=aA7UaS67eZs">
    <img src="https://dcgi.felk.cvut.cz/~futscdav/chunkmogrify/video_thumbnail.jpg" />
  </a>
</p>

Open an image using `File -> Image from File`. There is a sample image provided to check
functionality.  
  
Mask painting:  
Left click paints, right click unpaints. Mouse wheel controls the size of the brush.  

Projection:  
Input a number of steps (100 or 200 is ok, 500 is max before LR goes to 0 currently) and press
`Projection Steps`. Wait until projection finishes, you can observe the global image view by choosing
output mode `Projection Only` during this process. To fine-tune, you can perform a small number of
`Pivotal Tuning` steps.  

Editing:  
To add an edit, click the double arrow down icon in the Attribute Editor on the left side. Choose
the type of edit (W, S, Styleclip), the direction of the edit, and drag the sliders to change the
currently masked region. Usually it's necessary to increase the `multiplier` before noticeable
changes are reflected via the `direction` slider.  

Multiple different edits can be composed on top of each other at the same time. Their order
is largely irrelevant. Currently in the default mode, only one region is being edited, and so
all selected edits apply to the same region. If you would like to change the region, you can
`Freeze` the current image, and perform a new projection, but you will lose the ability to change
existing edits.  

To save the current image, click the `Save Current Image` button. If the `Unalign` checkbox is 
active, the program will attempt to compose the aligned face back into the original image. Saved
images can be found in the `SavedImages` directory by default. This can be changed in `_config.yaml`.

## Keyboard shortcuts

Current keyboard shortcuts include:  
  
Show/Hide mask          :: Alt+M  
Toggle mask painting    :: Alt+N  

## W-space editing
Source for some of the basic directions:  
(https://twitter.com/robertluxemburg/status/1207087801344372736)  
  
To add your own directions, save them in a numpy pickle format as a (num_ws, 512) or (1, 512)
format and specify their path in `w_directions.py`.

## Style-space editing (S space edits)  
Source:  
StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation  
(https://arxiv.org/abs/2011.12799)  
(https://github.com/betterze/StyleSpace)  
  
The presets can be found in `s_presets.py`, some were taken directly from the paper, others
I found by manual exploration. You can perform similar exploration by choosing the `Custom`
preset once you have a projection.

## StyleCLIP editing
Source:  
StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery  
(https://arxiv.org/abs/2103.17249)  
(https://github.com/orpatashnik/StyleCLIP)  
  
Pretrained models taken from (https://github.com/orpatashnik/StyleCLIP/blob/main/utils.py) and
manually removed the decoder from the state dict, since it's not used and takes up majority of
file size.

## PTI Optimization
Source:  
Pivotal Tuning for Latent-based Editing of Real Images  
(https://arxiv.org/abs/2106.05744)  
  
This method allows you to match the target photo very closely, while retaining editing capacities.  
  
It's often good to run 30-50 iterations of PTI to get very close matching of the source image,
which won't cause a very noticeable drop in the editing capabilities.


## Attribution
This repository makes use of code provided by the various repositories linked above, plus
additionally code from:  

styleganv2-ada-pytorch (https://github.com/NVlabs/stylegan2-ada-pytorch)  
poisson-image-editing (https://github.com/PPPW/poisson-image-editing) for optional support
of idempotent blend (slow implementation of blending that only changes the masked part which
can be accessed by uncommenting the option in `synthesis.py`)  

## Citation

If you find this code useful for your research, please cite the arXiv submission linked above.  
