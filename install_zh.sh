pip install torch torchvision torchaudio xformers --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

packages=(
opencv-python-headless
scipy
imageio
rawpy
kornia
)

for((i=0;i<${#packages[@]};i++))
do
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
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
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
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
    pip install ${packages[i]} -i https://pypi.tuna.tsinghua.edu.cn/simple
done