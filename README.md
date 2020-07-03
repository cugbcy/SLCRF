# SLCRF
SLCRF: Subspace Learning with Conditional Random Field for Hyperspectral Image Classification

By Yun Cao, Jie Mei, Yuebin Wang, Liqiang Zhang, Junhuan Peng, Bing Zhang, Lihua Li, Yibo Zheng

Train proprcess

1.Training the HSI with the 3DCAE, and obtaining the subspace feature representation.

2.Running the CAE_CE, and updating the weights and bias of CAE.

3.Running SLCRF, and updating the subspace feature representation.

4.Fine-tuning the CAE_CE with the new subspace feature representation.
