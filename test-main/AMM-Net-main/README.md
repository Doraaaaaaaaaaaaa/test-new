# AMM-Net
* PyTorch 1.6.0 implementation of the following paper. This version uses Swin Transformer as the backbone network, and you can replace it with other backbone network.  
* Test.py for model test.  
* Pretrained model weight available at: https://pan.quark.cn/s/b30129eab7bb  
* If you want to retrain the aesthetic attribute pre-trained model, you need download AADB database.  

# Citation
If you find our work is useful for your project or research, please consider citing our paper:  
@ARTICLE{AMM-Net,  
author={Li, Leida and Zhu, Tong and Chen, Pengfei and Yang, Yuzhe and Li, Yaqian and Lin, Weisi},  
journal={IEEE Transactions on Circuits and Systems for Video Technology},  
title={Image Aesthetics Assessment With Attribute-Assisted Multimodal Memory Network},  
volume={33},  
number={12},  
pages={7413-7424},  
doi={10.1109/TCSVT.2023.3272984}}  

# AMM-Net + CLIP Attribute for AVA Aesthetic Assessment

本项目基于 AMM-Net 做了适配与重构，用于 **AVA 图像美学评估**。  
当前版本支持：

- 图像主干分支：Swin Transformer
- 文本分支：BERT + GRU
- 属性分支：CLIP prompt-bank attribute encoder
- 多模态融合：MIMN
- 标签形式：AVA 评分分布（10 维）
- 指标：EMD / Accuracy / PLCC / SRCC
- 数据输入：`train.csv` / `val.csv` / `test.csv` + AVA 图片目录