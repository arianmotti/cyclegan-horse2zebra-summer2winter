# cyclegan-horse2zebra-summer2winter

1. Project structure

Suggested layout:

```text
.
├── notebooks/
│   ├── cyclegan_horse2zebra_64.ipynb
│   └── cyclegan_summer2winter_64.ipynb
├── src/
│   ├── models.py            # ResNet generators + PatchGAN discriminators
│   ├── dataset.py           # unpaired dataset loader
│   └── utils.py             # training loop helpers, plotting, etc.
├── results/
│   ├── horse2zebra/
│   │   ├── losses.png                   # G/cycle/id + D_A/D_B curves
│   │   ├── epoch_times.png              # time per epoch
│   │   └── qualitative_e1_e10_e20.png   # translations at epochs 1, 10, 20
│   └── summer2winter/
│       ├── losses.png
│       ├── epoch_times.png
│       └── qualitative_e1_e10_e20.png
├── data/                                # NOT tracked
├── .gitignore
└── README.md
.gitignore should ignore:

text
Copy code
data/
*.pth
*.pt
results/
.ipynb_checkpoints/
2. Environment
Tested with:

Python 3.11

PyTorch + CUDA (GTX 1050 Ti, 4 GB)

torchvision

numpy

matplotlib

Pillow

Example setup:

bash
Copy code
conda create -n cyclegan64 python=3.11
conda activate cyclegan64

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib pillow
3. Datasets
Place the downloaded CycleGAN datasets under data/:

text
Copy code
data/
├── horse2zebra/
│   ├── trainA/   # horses
│   ├── trainB/   # zebras
│   ├── testA/
│   └── testB/
└── summer2winter_yosemite/
    ├── trainA/   # summer
    ├── trainB/   # winter
    ├── testA/
    └── testB/
All images are:

resized to 64×64

converted to tensors

normalized using mean (0.5, 0.5, 0.5) and std (0.5, 0.5, 0.5) (so the network works in [-1, 1] with tanh output)

4. Model
4.1 Generators – ResNet-based
The generators G_AB and G_BA use a standard ResNet-style architecture, similar to the original CycleGAN paper:

initial c7s1-64 with reflection padding

two downsampling layers with stride-2 convolutions

a stack of residual blocks

two upsampling layers (transpose convolutions)

final c7s1-3 with tanh activation

To keep memory usage manageable at 64×64, the number of residual blocks is reduced (e.g. 6 blocks instead of 9 for 256×256).

4.2 Discriminators – PatchGAN
The discriminators D_A and D_B are PatchGAN discriminators:

Input: 3-channel 64×64 images

Several convolutional layers with increasing channels

Output: a spatial map of real/fake scores corresponding to ~70×70 receptive fields

Normalization: InstanceNorm

Activation: LeakyReLU

4.3 Losses & hyperparameters
The training objective consists of three main losses:

GAN loss for both directions:

G_AB vs D_B (A→B)

G_BA vs D_A (B→A)

Cycle-consistency loss with weight:

lambda_cycle = 10

Identity loss with weight:

lambda_id = 5

Optimizer and hyperparameters:

Optimizer: Adam

Learning rate: lr = 0.0002

Betas: β1 = 0.5, β2 = 0.999

Batch size: 1

Image size: 64×64

For improved stability, an image buffer (history of previously generated images) can be used when training the discriminators, so they do not only see the very latest generator outputs.

5. Training – horse2zebra
Notebook: notebooks/cyclegan_horse2zebra_64.ipynb

5.1 Settings
Dataset: horse2zebra

Epochs: 20

Batch size: 1

Image size: 64×64

Optimizer: Adam as above

5.2 Training time
On a GTX 1050 Ti (4 GB):

Time per epoch ≈ ~500 seconds

Total training time for 20 epochs ≈ ~166 minutes

5.3 Logged curves and figures
The notebook logs and plots:

Generator total loss, cycle loss, and identity loss vs epoch:

All three decrease over time and flatten as the model converges.

Discriminator losses (D_A, D_B) vs epoch:

Both start around ~0.25–0.27 and decrease to ~0.16 with moderate oscillations.

Epoch time vs epoch:

Approximately constant around 490–510 seconds.

These plots are saved under:

text
Copy code
results/horse2zebra/losses.png
results/horse2zebra/epoch_times.png
5.4 Qualitative results
For a randomly chosen horse image (Input A) and a zebra image (Input B) from the test set, we visualize:

A → B and B → A at epochs 1, 10, and 20:

text
Copy code
Input A      | A→B (e1) | A→B (e10) | A→B (e20)
Input B      | B→A (e1) | B→A (e10) | B→A (e20)
Saved as:

text
Copy code
results/horse2zebra/qualitative_e1_e10_e20.png
Observations:

At epoch 1, outputs are blurry and noisy; the global structure is roughly preserved but stripes/textures are not convincing.

At epoch 10, clear zebra stripes start to appear in A→B, and horses become more realistic in B→A, while backgrounds and shapes are preserved.

At epoch 20, the style transfer is more consistent, but some details remain blurred due to the low 64×64 resolution.

6. Training – summer2winter_yosemite
Notebook: notebooks/cyclegan_summer2winter_64.ipynb

6.1 Settings
Dataset: summer2winter_yosemite

Epochs: 20

Same hyperparameters and architecture as horse2zebra.

Again, images resized to 64×64 for efficiency.

6.2 Training time
On the same hardware:

Time per epoch ≈ 305–309 seconds

Total training time for 20 epochs ≈ ~102 minutes

6.3 Logged curves & figures
The notebook produces:

Generator loss curves (G total, cycle, identity).

Discriminator loss curves for D_A (summer) and D_B (winter).

Epoch time plot.

Typical behavior from the reference run:

G loss decreases from around 8.7 to about 5.7.

cycle loss decreases from about 5.3 to around 2.9.

identity loss decreases from about 2.4 to around 1.3 and then stabilizes.

D_A loss and D_B loss decrease from roughly 0.26–0.27 to 0.10–0.12.

Saved under:

text
Copy code
results/summer2winter/losses.png
results/summer2winter/epoch_times.png
6.4 Qualitative results
Again, we pick one test image from each domain:

Input A = summer

Input B = winter

Then we visualize:

A → B and B → A at epochs 1, 10, 20:

text
Copy code
Input A      | A→B (e1) | A→B (e10) | A→B (e20)
Input B      | B→A (e1) | B→A (e10) | B→A (e20)
Saved as:

text
Copy code
results/summer2winter/qualitative_e1_e10_e20.png
Qualitative observations:

Epoch 1: outputs are noisy with weak seasonal changes.

Epoch 10: images in A→B start to show snow, colder tones and winter-like lighting; B→A images become warmer with less snow.

Epoch 20: seasonal style transfer is clearly visible, but the low 64×64 resolution limits sharpness and fine details.

7. How to run
Download and extract horse2zebra and summer2winter_yosemite under data/ in the structure described above.

Install the environment.

Start Jupyter:

bash
Copy code
jupyter notebook
For horse2zebra:

Open notebooks/cyclegan_horse2zebra_64.ipynb

Run all cells

For summer2winter:

Open notebooks/cyclegan_summer2winter_64.ipynb

Run all cells

Each notebook:

builds and trains the models,

logs and plots all requested losses,

prints epoch training times,

saves qualitative examples and checkpoints (.pth).

8. Notes
This implementation is tuned for a single mid-range GPU and low resolution (64×64).
For higher visual quality, retrain with 128×128 or 256×256 images and more epochs.

The code and experiments were created as part of a Deep Generative Models course assignment.

Feel free to reuse or extend the code for learning and research.
