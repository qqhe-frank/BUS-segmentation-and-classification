# BUS-segmentation-and-classification
Multi-task learning for segmentation and classification of breast tumors from ultrasound images

# Abstract
Segmentation and classification of breast tumors are critical components of breast ultrasound (BUS) computer-aided diagnosis (CAD), which significantly improves the diagnostic accuracy of breast cancer. However, the characteristics of tumor regions in BUS images, such as non-uniform intensity distributions, ambiguous or missing boundaries, and varying tumor shapes and sizes, pose significant challenges to automated segmentation and classification solutions. Many previous studies have proposed multi-task learning methods to jointly tackle tumor segmentation and classification by sharing the features extracted by the encoder. Unfortunately, this often introduces redundant or misleading information, which hinders effective feature exploitation and adversely affects performance. To address this issue, we present ACSNet, a novel multi-task learning network designed to optimize tumor segmentation and classification in BUS images. The segmentation network incorporates a novel gate unit to allow optimal transfer of valuable contextual information from the encoder to the decoder. In addition, we develop the Deformable Spatial Attention Module (DSAModule) to improve segmentation accuracy by overcoming the limitations of conventional convolution in dealing with morphological variations of tumors. In the classification branch, multi-scale feature extraction and channel attention mechanisms are integrated to discriminate between benign and malignant breast tumors. Experiments on two publicly available BUS datasets demonstrate that ACSNet not only outperforms mainstream multi-task learning methods for both breast tumor segmentation and classification tasks, but also achieves state-of-the-art results for BUS tumor segmentation.
# framwork:
![network](https://github.com/user-attachments/assets/8e422aad-c244-488c-b099-645c3a60aab7)

Thanks to the following projects open source:
1. https://github.com/msracver/Deformable-ConvNets (DCN)
2. https://github.com/xorangecheng/GlobalGuidance-Net
3. https://github.com/mroussak/BUS_Deep_Learning、https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch

# Note
DCN:Deformable Convolution (DCN) code from https://github.com/msracver/Deformable-ConvNets

If you have any questions, please contact:
20210217h@gmail.com
