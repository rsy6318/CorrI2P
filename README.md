## CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence  
[arxiv](https://arxiv.org/pdf/2207.05483.pdf)  [TCSVT](https://ieeexplore.ieee.org/document/9900350)  
Accepted by IEEE TCSVT  
![Correspondence](pic/correspondence_ours.png)  
### Data  
#### KITTI
Here we would provide KITTI prepared.  
You can download it [here](https://tjueducn-my.sharepoint.com/:f:/g/personal/rsy6318_tju_edu_cn/Ejuy4n_OeuFPkayDWnOwRmgBRnR2z_pltD2uv0F6LHYN_Q?e=7506Ug).  
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
We would provide the script for preparing NuScenes dataset in **nuScenes_script** folder (reffer to [DeepI2P](https://github.com/lijx10/DeepI2P)). They can be used to generate nuscenes dataset. 
### Usage
Install required lib as SO-Net or [DeepI2P](https://github.com/lijx10/DeepI2P/tree/main/models/index_max_ext).
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
@ARTICLE{CORRI2P,
  author={Ren, Siyu and Zeng, Yiming and Hou, Junhui and Chen, Xiaodong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3208859}}
```

### Acknowledgement
We thank the authors of [DeepI2P](https://github.com/lijx10/DeepI2P) for their public code.

If you want to use our code, please cite our work.

