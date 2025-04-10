pip install torch torchvision torchaudio xformers

packages=(
opencv-python-headless
scipy
imageio
rawpy
kornia
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]}
done

packages=(
diffusers[torch]
accelerate
transformers
einops
timm
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]}
done

packages=(
wandb
matplotlib
omegaconf
fvcore
python-dotenv
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]}
done