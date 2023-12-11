# ORNet: Object Refinement Network Based on Dual-Branch and Multi-Feature Enhancement for Saliency Detection

## Overview
In recent years, CNN-based salient object detection methods have shown significant performance improvements. Recent research on salient object detection (SOD) has primarily focused on pixel-level classification using encoder-decoder models based on fully convolutional networks (FCN). However, existing CNN-based saliency detection networks have a drawback of overemphasizing pixel-level information of objects while neglecting background information, leading to limited capability in refining the saliency of object boundaries and capturing fine-grained details. To address this issue, we introduce ORNet: Object Refinement Network, to achieve superior saliency detection. We propose a novel dual-branch module consisting of a foreground module (FM) and a background module (BM) to extract foreground and background information, respectively. Each branch adopts multiple feature enhancement strategies. Specifically, this strategy includes the Object Refinement Module (ORM) for fusing multi-scale contextual features to enhance the saliency of objects. Additionally, we propose the Fourier Edge Extractor (FEE) module to extract boundary features from the frequency domain, enhancing the clarity of object boundaries. We introduce the Global Context Block (GCB) at the final network layer to extract high-level semantic features using pooling operations, which aids in precise object localization through the dual-branch module. Furthermore, to further enhance object boundaries, we design the Boundary Diffusion Loss (BDL) function to guide the network's attention to the often overlooked boundary information. We choose the EfficientNet architecture as the backbone network due to its computational efficiency. Finally, extensive experiments on popular benchmark datasets demonstrate the competitive performance of ORNet. Both the codes and results is available at [https://github.com/CKCAHU/ORNet](https://github.com/CKCAHU/ORNet).

## Saliency Map
Saliency maps: [Baidu Drive](https://pan.baidu.com/s/18l3GJDwX3hXyeVgTSxu9Qw?pwd=gm55) [Google Drive](https://drive.google.com/file/d/1iNJPTEHyYtX_qxuMlaKog2UOgYolxi0q/view?usp=drive_link)
