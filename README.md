NOTICE
#Important environment setting
python 3.8
torch                       2.1.1+cu118
torchaudio                  2.1.1+cu118
torchmetrics                1.1.1
torchvision                 0.16.1+cu118

#Steps for installing environment
step1. conda create -n dual_cam python=3.8
step2. module load cuda/11.8
step3. conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
step4. pip install -r requirements.txt
step5. pip install submodules/simple-knn
step6. pip install submodules/diff-gaussian-rasterization
step7. pip install submodules/fused-ssim
step8. mkdir -p mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P mast3r/checkpoints/
