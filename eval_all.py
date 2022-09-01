import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
from network import DenseI2P
from kitti_pc_img_dataloader import kitti_pc_img_dataset
#from loss2 import kpt_loss, kpt_loss2, eval_recall
import datetime
import logging
import math
import numpy as np
import options

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=25, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=12, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=24, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='/home/siyu_ren/kitti_dataset/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_point', type=int, default=40960, metavar='num_point',
                        help='point cloud size to train')
    parser.add_argument('--num_workers', type=int, default=6, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=300, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.01, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=0, metavar='P_Rz_amplitude',
                        help='')


    parser.add_argument('--save_path', type=str, default='./log', metavar='save_path',
                        help='path to save log and model')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='dist_thres',
                        help='')    
    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')  
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')                           
    args = parser.parse_args()

    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.num_point,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,is_front=False)
    #assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    #trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,drop_last=True,num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    
    opt=options.Options()
    
    model=DenseI2P(opt)
    #model.load_state_dict(torch.load('./log/2021-11-29 19:30:59.134507/mode_best_t.t7'))
    #model.load_state_dict(torch.load('./log/2022-01-03 14:30:04.051421/mode_last.t7'))

    model.load_state_dict(torch.load('./log_xy_40960_128/dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f/mode_last.t7'%(args.dist_thres,args.pos_margin,args.neg_margin)))
    model=model.cuda()
    save_path='result_all_dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(args.dist_thres,args.pos_margin,args.neg_margin)
    try:
        os.mkdir(save_path)
    except:
        pass
    with torch.no_grad():
        for step,data in enumerate(testloader):
            model.eval()
            img=data['img'].cuda()
            pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()
            img_mask=data['img_mask'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()
            pc_feature=torch.cat((intensity,sn),dim=1)
            img_feature,pc_feature,img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)
            
            np.save(os.path.join(save_path,'img_%d.npy'%(step)),img.cpu().numpy())
            np.save(os.path.join(save_path,'pc_%d.npy'%(step)),pc.cpu().numpy())
            np.save(os.path.join(save_path,'pc_score_%d.npy'%(step)),pc_score.data.cpu().numpy())
            np.save(os.path.join(save_path,'pc_mask_%d.npy'%(step)),pc_mask.data.cpu().numpy())
            np.save(os.path.join(save_path,'K_%d.npy'%(step)),K.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_mask_%d.npy'%(step)),img_mask.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_score_%d.npy'%(step)),img_score.data.cpu().numpy())
            np.save(os.path.join(save_path,'img_feature_%d.npy'%(step)),img_feature.data.cpu().numpy())
            np.save(os.path.join(save_path,'pc_feature_%d.npy'%(step)),pc_feature.data.cpu().numpy())
            np.save(os.path.join(save_path,'P_%d.npy'%(step)),P.data.cpu().numpy())
            
            