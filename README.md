## CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence  
[arxiv](https://arxiv.org/abs/2207.05483)  
![Correspondence](pic/correspondence_ours.png)  
### Data  
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
@article{ren2022corri2p,
  title={CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence},
  author={Ren, Siyu and Zeng, Yiming and Hou, Junhui and Chen, Xiaodong},
  journal={arXiv preprint arXiv:2207.05483},
  year={2022}
}
```

### Acknowledgement
We thank the authors of [DeepI2P](https://github.com/lijx10/DeepI2P) for their public code.

If you want to use our code, please cite our work.

