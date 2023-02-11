# Fusion_HCT

Python demos for the paper "Joint Classification of Hyperspectral and LiDAR Data Using a Hierarchical CNN and Transformer".

Using the code should cite the following paper:

G. Zhao, Q. Ye, L. Sun, Z. Wu, C. Pan, and B. Jeon, "Joint Classification of Hyperspectral and LiDAR Data Using a Hierarchical CNN and Transformer", in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-16, 2023, Art no. 5500716, doi: 10.1109/TGRS.2022.3232498.

@ARTICLE{9999457,
  author={Zhao, Guangrui and Ye, Qiaolin and Sun, Le and Wu, Zebin and Pan, Chengsheng and Jeon, Byeungwoo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Joint Classification of Hyperspectral and LiDAR Data Using a Hierarchical CNN and Transformer}, 
  year={2023},
  volume={61},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2022.3232498}
}

L. Sun, G. Zhao, Y. Zheng and Z. Wu, "Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5522214,  doi: 10.1109/TGRS.2022.3144158.
  
@ARTICLE{9684381,  
    author={Sun, Le and Zhao, Guangrui and Zheng, Yuhui and Wu, Zebin},  
    journal={IEEE Transactions on Geoscience and Remote Sensing},   
    title={Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification},   
    year={2022}, 
    volume={60},  
    number={},  
    pages={1-14},  
    doi={10.1109/TGRS.2022.3144158}
}

Feel free to contact us if there is anything we can help. Thanks for your support!

cs_zhaogr@nuist.edu.cn 

# Description.

  The joint use of multisource remote-sensing (RS) data for Earth observation missions has drawn much attention. Although the fusion of several data sources can improve the accuracy of land-cover identification, many technical obstacles, such as disparate data structures, irrelevant physical characteristics, and a lack of training data, exist. In this article, a novel dual-branch method, consisting of a hierarchical convolutional neural network (CNN) and a transformer network, is proposed for fusing multisource heterogeneous information and improving joint classification performance. First, by combining the CNN with a transformer, the proposed dual-branch network can significantly capture and learn spectral–spatial features from hyperspectral image (HSI) data and elevation features from light detection and ranging (LiDAR) data. Then, to fuse these two sets of data features, a cross-token attention (CTA) fusion encoder is designed in a specialty. The well-designed deep hierarchical architecture takes full advantage of the powerful spatial context information extraction ability of the CNN and the strong long-range dependency modeling ability of the transformer network based on the self-attention (SA) mechanism. Four standard datasets are used in experiments to verify the effectiveness of the approach. The experimental results reveal that the proposed framework can perform noticeably better than state-of-the-art methods. 
