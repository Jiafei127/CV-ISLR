

pip install mmcv-full==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html


pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html


tsm_imagenet-pretrained-r50_8xb16-1x1x15-50e_www.py


## TSM

python tools/train.py configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x15-50e_www.py  --seed=0 --deterministic

python tools/train.py configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www.py  --seed=0  

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia


pip install -U openmim

mim install mmengine


pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

mim install mmdet

mim install mmpose

pip install -v -e .

######################################################

训练


conda activate openmmlab
configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www.py

depth
configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-www.py

# 4卡训练模型

TSM


CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x15-50e_www.py 4 --seed=0  
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_test.sh  configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x15-50e_www.py  /mnt/mydisk1/wangfei/WWW/mmaction2/work_dirs/tsm_imagenet-pretrained-r50_8xb16-1x1x15-50e_www/best_acc_top1_epoch_41.pth  4  --dump result_tsm.pkl



tiny
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www.py 4 --seed=0 


base

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www.py 4 --seed=0 


large 
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_www.py 4 --seed=0 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 bash tools/dist_train.sh configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-www.py 8 --seed=1234

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_test.sh /mnt/mydisk1/wangfei/www/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-www_rgb.py /mnt/mydisk1/wangfei/www/mmaction2/work_dirs/uniformerv2-base-p16-res224_clip_8xb32-www_rgb/epoch_55.pth  4 --dump result.pkl




# 单卡测试模型
python tools/train.py configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-www.py



bash tools/dist_test.sh configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www.py ./work_dirs/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www/best_acc_top1_epoch_30.pth 4 --dump result.pkl


./work_dirs/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www/20241228_220454

tiny
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_test.sh configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-80e_www.py /mnt/mydisk1/wangfei/WWW/mmaction2/work_dirs/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_www/best_acc_top1_epoch_30.pth  4 --dump result_swin-tiny.pkl