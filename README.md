## CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence [[arxiv](https://arxiv.org/pdf/2207.05483.pdf),  [TCSVT](https://ieeexplore.ieee.org/document/9900350)]   
Accepted by IEEE TCSVT  

### News !!!   
* The download link of preprocessed datasets is temporarily unavailable, and we are in the process of finding a new cloud storage solution.    
* Our new work [**Self-supervised Learning of LiDAR 3D Point Clouds via 2D-3D Neural Calibration**](https://arxiv.org/abs/2401.12452) achieves a higher accuracy, and we refer the readers to follow! [[Code](https://github.com/Eaphan/NCLR)]

![Correspondence](pic/correspondence_ours.png)  
### Data  
#### KITTI
Here we provide KITTI prepared.  
You can download it [here](https://portland-my.sharepoint.com/:u:/g/personal/siyuren2-c_my_cityu_edu_hk/EY_3Cwr3PhZHiNj_ijDPIp0BZx23H9T1J-wrmd6gqbgykw?e=4quHFS).  
Unzip these files, and the directory is as follows:  
```
kitti
-calib
--00
--01
...
-sequences
--00
--01
...
```
#### NuScenes  
Here we provide nuScenes prepared.  
You can download it [here](https://portland-my.sharepoint.com/:f:/g/personal/siyuren2-c_my_cityu_edu_hk/EhCjMm7W95pEvd0W0rMZduUBvwsUfzIYT9r3opBULD8O4g?e=0XIfhq).  
We also provide the script for preparing NuScenes dataset in **nuScenes_script** folder (reffer to [DeepI2P](https://github.com/lijx10/DeepI2P)). They can be used to generate nuscenes dataset. 
### Usage
Install required lib as SO-Net or [DeepI2P](https://github.com/lijx10/DeepI2P/tree/main/models/index_max_ext).  
[Indexmax](https://github.com/lijx10/SO-Net/tree/master/models/index_max_ext)
#### Train
```sh
python train.py
```
#### Test
```sh
python eval_all.py
python cal_error_all.py
python analysis.py
```
Note: There would be lots of intermediate results, please leave enough storage space.  
  
### Citation
```bibtex
@article{ren2022corri2p,
  title={Corri2p: Deep image-to-point cloud registration via dense correspondence},
  author={Ren, Siyu and Zeng, Yiming and Hou, Junhui and Chen, Xiaodong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={33},
  number={3},
  pages={1198--1208},
  year={2022},
  publisher={IEEE}
}
```

### Acknowledgement
We thank the authors of [DeepI2P](https://github.com/lijx10/DeepI2P) for their public code.

If you want to use our code, please cite our work.



