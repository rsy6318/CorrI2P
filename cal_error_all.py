import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation
import multiprocessing
import argparse


def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))

    return t_diff,angles_diff

def get_diff(base_path,save_path,dist_thres,num,num_total):
    t_diff_set=[]
    angles_set=[]
    is_success_set=[]
    T_pred_set=[]
    T_gt_set=[]

    img_score_set=np.load(os.path.join(base_path,'img_score_%d.npy'%(num)))
    pc_score_set=np.load(os.path.join(base_path,'pc_score_%d.npy'%(num)))
    img_feature_set=np.load(os.path.join(base_path,'img_feature_%d.npy'%(num)))
    pc_feature_set=np.load(os.path.join(base_path,'pc_feature_%d.npy'%(num)))
    pc_set=np.load(os.path.join(base_path,'pc_%d.npy'%(num)))
    P_set=np.load(os.path.join(base_path,'P_%d.npy'%(num)))
    K_set=np.load(os.path.join(base_path,'K_%d.npy'%(num)))

    for i in range(img_score_set.shape[0]):
        

        img_score=img_score_set[i]
        pc_score=pc_score_set[i]
        img_feature=img_feature_set[i]
        pc_feature=pc_feature_set[i]
        pc=pc_set[i]
        P=P_set[i]
        K=K_set[i]

        img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
        img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
        img_xy=np.concatenate((img_x,img_y),axis=0)

        #print(img_xy[:,22,1])

        img_xy_flatten=img_xy.reshape(2,-1)
        img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
        img_score_flatten=img_score.squeeze().reshape(-1)


        '''print(img_xy_flatten[:,10])
        print(img_feature_flatten[:,10])
        print(img_feature[:,int(img_xy_flatten[1,10]),int(img_xy_flatten[0,10])])'''



        img_xy_flatten_sel=img_xy_flatten[:,img_score_flatten>0.9]
        img_feature_flatten_sel=img_feature_flatten[:,img_score_flatten>0.9]
        img_score_flatten_sel=img_score_flatten[img_score_flatten>0.9]

        #print(np.min(img_score_flatten_sel))



        pc_sel=pc[:,pc_score.squeeze()>0.9]
        pc_feature_sel=pc_feature[:,pc_score.squeeze()>0.9]
        pc_score_sel=pc_score.squeeze()[pc_score.squeeze()>0.9]



        #dist=np.sum((np.expand_dims(pc_feature_sel,axis=2)-np.expand_dims(img_feature_flatten_sel,axis=1))**2,axis=0)
        dist=1-np.sum(np.expand_dims(pc_feature_sel,axis=2)*np.expand_dims(img_feature_flatten_sel,axis=1),axis=0)
        sel_index=np.argsort(dist,axis=1)[:,0]
        img_xy_pc=img_xy_flatten_sel[:,sel_index]

        try:
            is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                        iterationsCount=500,
                                                        reprojectionError=dist_thres,
                                                        flags=cv2.SOLVEPNP_EPNP,
                                                        distCoeffs=None)
        except:
            print(num*img_score_set.shape[0]+i,'has problem!')
            print('pc shape',pc_sel.shape,'img shape',img_xy_pc.shape)
            assert False
        R,_=cv2.Rodrigues(R)
        '''print(R)
        print(t)
        print(P)
        print(is_success)'''
        T_pred=np.eye(4)
        T_pred[0:3,0:3]=R
        T_pred[0:3,3:]=t
        #print(T_pred)

        t_diff,angles_diff=get_P_diff(T_pred,P)
        t_diff_set.append(t_diff)
        angles_set.append(angles_diff)
        is_success_set.append(is_success)
        T_pred_set.append(T_pred)
        T_gt_set.append(P)

        print(num,'/',num_total,' ',i,'/',img_score_set.shape[0],'RTE: ',t_diff,' RRE: ',angles_diff)

    t_diff_set=np.array(t_diff_set)
    angles_set=np.array(angles_set)
    is_success_set=np.array(is_success_set)
    T_pred_set=np.array(T_pred_set)
    T_gt_set=np.array(T_gt_set)

    np.save(os.path.join(save_path,'t_error_%d.npy'%num),t_diff_set)
    np.save(os.path.join(save_path,'angle_error_%d.npy'%num),angles_set)
    np.save(os.path.join(save_path,'is_success_%d.npy'%num),is_success_set)
    np.save(os.path.join(save_path,'T_pred_%d.npy'%num),T_pred_set)
    np.save(os.path.join(save_path,'T_gt_%d.npy'%num),T_gt_set)
    #print('t_diff',t_diff)
    #print('t_diff',t_diff)
    #print('R_diff',angles_diff)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Registration')

    parser.add_argument('--train_dist_thres', type=float, default=1, metavar='train_dist_thres',
                            help='')
    parser.add_argument('--test_dist_thres', type=float, default=1, metavar='test_dist_thres',
                            help='')
    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                            help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                            help='')
    args = parser.parse_args()


    base_path='./result_all_dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f/'%(args.train_dist_thres,args.pos_margin,args.neg_margin)

    num_total=len(os.listdir(base_path))//10
    save_path='./result_error_%0.2f_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f/'%(args.train_dist_thres,args.test_dist_thres,args.pos_margin,args.neg_margin)
    try:
        os.mkdir(save_path)
    except:
        pass

    thres_size=8
    num_iter=num_total//thres_size
    for k in range(num_iter):

        threads=[]
        for num in range(int(k*thres_size),int(k*thres_size+thres_size)):
            threads.append(multiprocessing.Process(target=get_diff,args=([base_path,save_path,args.test_dist_thres,num,num_total])))

        for thread in threads:
            thread.start()
        for i, thread in enumerate(threads):
            thread.join()

    threads = []
    for num in range(int(num_total//thres_size*thres_size+thres_size),int(num_total)):
        threads.append(multiprocessing.Process(target=get_diff, args=([num])))

    for thread in threads:
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()

    '''for num in range(num_total):
        get_diff(num)'''
    