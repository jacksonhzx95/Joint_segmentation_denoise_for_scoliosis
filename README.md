# Joint_segmentation_denoise_for_scoiliosis
This is the implementation of paper: Z. Huang et al., "Joint Spine Segmentation and Noise Removal From Ultrasound Volume Projection Images With Selective Feature Sharing," in IEEE Transactions on Medical Imaging, vol. 41, no. 7, pp. 1610-1624, July 2022, doi: 10.1109/TMI.2022.3143953.
# Requirements
mmcv python3.8 pytorch1.7
# Enviroment 
pip install-e .
# test the code
python inference.py --config {config file path} {checkpoint file path} --show --show_mask {save path for segmentation result} --show_denoise --show_dir {save path for denoised image}
# training
python train.py {config file path} --work-dir {the path to save checkpoints, training log} --load-from [optianal] {for finetune or load pretrain model}

# more details can follow the document of mmsegmentation
https://mmsegmentation.readthedocs.io/en/latest/
