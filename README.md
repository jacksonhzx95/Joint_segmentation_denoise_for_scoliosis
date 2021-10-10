# Joint_segmentation_denoise_for_scoiliosis
This is the official implementation of paper: Joint Spine Segmentation and Noise Removal from Ultrasound Volume Projection Images with Selective Feature Sharing
# Requirements
mmcv python3.8 pytorch1.7
# enviroment 
pip install-e .
# test the code
python inference.py --config {config file path} {checkpoint file path} --show --show_mask {save path for segmentation result} --show_denoise --show_dir {save path for denoised image}
# training
python train.py {config file path} --work-dir {the path to save checkpoints, training log} --load-from [optianal] {for finetune or load pretrain model}

# more details can follow the document of mmsegmentation
https://mmsegmentation.readthedocs.io/en/latest/
