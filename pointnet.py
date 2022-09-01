import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PCUpSample(nn.Module):
    def __init__(self,in_channel,mlp,last_norm_activate=True,k=16):
        super(PCUpSample,self).__init__()
        self.mlp_convs=nn.ModuleList()
        self.mlp_bns=nn.ModuleList()
        last_channel=in_channel
        self.last_norm_activate=last_norm_activate
        self.k=k
        for out_channel in mlp[0:-1]:
            self.mlp_convs.append(nn.Conv1d(last_channel,out_channel,1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel=out_channel

        if self.last_norm_activate:
            self.mlp_convs.append(nn.Conv1d(mlp[-2],mlp[-1],1))
            self.mlp_bns.append(nn.BatchNorm1d(mlp[-1]))
        else:
            self.mlp_convs.append(nn.Conv1d(mlp[-2],mlp[-1],1))



    def forward(self,xyz1,xyz2,features1=None,features2=None):
        '''
        xyz1: denstination points
        xyz2: downsampled points
        features1: densitination features
        features2: downsampled features
        '''
        B,_,N=xyz1.shape
        
        xyz1=xyz1.permute(0,2,1)
        xyz2=xyz2.permute(0,2,1)

        
        features2=features2.permute(0,2,1)

        dists=square_distance(xyz1,xyz2)
        dists,idx=dists.sort(dim=-1)
        dists,idx=dists[:,:,:self.k], idx[:,:,:self.k]

        dist_recip=1.0/(dists+1e-8)
        
        norm=torch.sum(dist_recip,dim=2)
        weight=dist_recip/norm.unsqueeze(-1)
        #print(weight.size(),index_points(features2, idx).size())
        interpolated_features = torch.sum(index_points(features2, idx) * weight.view(B, N, self.k, 1), dim=2)

        if features1 is None:
            new_features=interpolated_features
        else:

            features1=features1.permute(0,2,1)
            new_features=torch.cat([features1,interpolated_features],dim=-1)

        new_features=new_features.permute(0,2,1)
       
        for i, conv in enumerate(self.mlp_convs[:-1]):
            bn=self.mlp_bns[i]
            new_features=F.relu(bn(conv(new_features)))
        if self.last_norm_activate:
            bn=self.mlp_bns[-1]
            conv=self.mlp_convs[-1]
            new_features=F.relu(bn(conv(new_features)))
        else:
            conv=self.mlp_convs[-1]
            new_features=conv(new_features)
        return new_features

def FPS(pc,k):
    '''
    pc:(B,C,N)
    return (B,C,k)
    '''
    device=pc.device
    B,C,N=pc.size()
    centroids = torch.zeros(B,k,dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device)*1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(k):
        centroids[:, i] = farthest
        centroid = pc[batch_indices, :, farthest].view(B, 3, 1)
        dist = torch.sum((pc-centroid)**2, -2)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return torch.gather(pc, dim=2, index=centroids.unsqueeze(1).repeat(1, C, 1))

'''def FPS(pc, k):
    xyz=pc.transpose(2,1)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, k, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   #(B, 1)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(k):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return torch.gather(pc, dim=2, index=centroids.unsqueeze(1).repeat(1, C, 1))
'''

def group(pc,kpts,k):
    B,C,N=pc.size()
    N2=kpts.size(-1)
    diff=torch.sum(torch.square(pc.unsqueeze(2)-kpts.unsqueeze(3)),dim=1)
    _,idx=torch.topk(diff,k,dim=2,largest=False)
    grouped_pc=torch.gather(pc.unsqueeze(2).repeat(1,1,N2,1),dim=3,index=idx.unsqueeze(1).repeat(1,3,1,1))
    grouped_pc=grouped_pc-kpts.unsqueeze(-1)
    return grouped_pc


def group_with_feature(pc,features,kpts,k):
    B,C,N=pc.size()
    C_feature=features.size(1)
    N2=kpts.size(-1)
    diff=torch.sum(torch.square(pc.unsqueeze(2)-kpts.unsqueeze(3)),dim=1)
    dist,idx=torch.topk(diff,k,dim=2,largest=False)
    #print(dist[0])
    grouped_pc=torch.gather(pc.unsqueeze(2).repeat(1,1,N2,1),dim=3,index=idx.unsqueeze(1).repeat(1,C,1,1))
    grouped_pc=grouped_pc-kpts.unsqueeze(-1)
    grouped_features=torch.gather(features.unsqueeze(2).repeat(1,1,N2,1),dim=3,index=idx.unsqueeze(1).repeat(1,C_feature,1,1))
    return grouped_pc, grouped_features

def group_only_feature(pc,features,kpts,k):
    B,C,N=pc.size()
    C_feature=features.size(1)
    N2=kpts.size(-1)
    diff=torch.sum(torch.square(pc.unsqueeze(2)-kpts.unsqueeze(3)),dim=1)
    _,idx=torch.topk(diff,k,dim=2,largest=False)
    #grouped_pc=torch.gather(pc.unsqueeze(2).repeat(1,1,N2,1),dim=3,index=idx.unsqueeze(1).repeat(1,C,1,1))
    #grouped_pc=grouped_pc-kpts.unsqueeze(-1)
    grouped_features=torch.gather(features.unsqueeze(2).repeat(1,1,N2,1),dim=3,index=idx.unsqueeze(1).repeat(1,C_feature,1,1))
    return  grouped_features

class attention_img2pc(nn.Module):
    def __init__(self,in_channel,mlp):
        super(attention_img2pc,self).__init__()
        self.mlp_convs=nn.ModuleList()
        self.mlp_bns=nn.ModuleList()
        last_channel=in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel,out_channel,1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel=out_channel
    def forward(self,img_global_feature,img_local_feature,pc_local_feature):
        N=pc_local_feature.size(2)
        B=pc_local_feature.size(0)
        C_img=img_local_feature.size(1)
        feature=torch.cat([pc_local_feature,img_global_feature.unsqueeze(-1).repeat(1,1,N)],dim=1)
        for i, conv in enumerate(self.mlp_convs):
            bn=self.mlp_bns[i]
            feature=F.relu(bn(conv(feature)))
        attention=F.softmax(feature,dim=1)      #(B,H*W,N)
        img_local_feature=img_local_feature.view(B,C_img,-1)        #(B,C,H*W)
        #print(img_local_feature.size(),attention.size())
        feature_fusion=torch.matmul(img_local_feature,attention)
        return feature_fusion


def angle(v1,v2):
    """
    v1:(B,3,N,K)
    v2:(B,3,N,K)
    """
    cross_prod=torch.cat((  v1[:,1:2,:,:]*v2[:,2:,:,:]-v1[:,2:,:,:]*v2[:,1:2,:,:],
                            v1[:,2:,:,:]*v2[:,0:1,:,:]-v1[:,0:1,:,:]*v2[:,2:,:,:],
                            v1[:,0:1,:,:]*v2[:,1:2,:,:]-v1[:,1:2,:,:]*v2[:,0:1,:,:]),dim=1)
    cross_prod_norm=torch.norm(cross_prod,dim=1,keepdim=True)
    dot_prod=torch.sum(v1*v2,dim=1,keepdim=True)
    return torch.atan2(cross_prod_norm,dot_prod)



class PointCloudEncoder(nn.Module):
    def __init__(self,k1=512,k2=64,s1=256,s2=32):
        super(PointCloudEncoder,self).__init__()
        self.k1=k1
        self.k2=k2
        self.s1=s1
        self.s2=s2
        self.conv1=nn.Sequential(nn.Conv2d(8,64,1),nn.BatchNorm2d(64),nn.ReLU(),
                                nn.Conv2d(64,64,1),nn.BatchNorm2d(64),nn.ReLU(),
                                nn.Conv2d(64,256,1),nn.BatchNorm2d(256),nn.ReLU())

        self.conv2=nn.Sequential(nn.Conv2d(256+3,256,1),nn.BatchNorm2d(256),nn.ReLU(),
                                nn.Conv2d(256,256,1),nn.BatchNorm2d(256),nn.ReLU(),
                                nn.Conv2d(256,512,1),nn.BatchNorm2d(512),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv1d(512,512,1),nn.BatchNorm1d(512),nn.ReLU(),
                                nn.Conv1d(512,512,1),nn.BatchNorm1d(512),nn.ReLU())
    def forward(self,xyz,features):
        points=[]
        out=[]
        #----------------------
        xyz1=FPS(xyz,self.k1)
        xyz1_grouped,features1_grouped=group_with_feature(xyz,features,xyz1,self.s1)    #(B,3,N,K) (B,C,N,K)

        sn1=features1_grouped[:,1:,:,:]          #(B,3,N,K)
        intensity1=features1_grouped[:,0:1,:,:]  #(B,1,N,K)

        sn1_center=sn1[:,:,:,0:1]
        xyz1_center=xyz1.unsqueeze(-1)
        d=xyz1_grouped-xyz1_center
        nr_d=angle(sn1_center,d)
        ni_d=angle(sn1,d)
        nr_ni=angle(sn1_center,sn1)
        d_norm=torch.norm(d,dim=1,keepdim=True)
        features1_grouped=torch.cat((nr_d,ni_d,nr_ni,d_norm,intensity1),dim=1)

        #print(xyz.size(),xyz1.size(),xyz1_grouped.size())
        features1=self.conv1(torch.cat((xyz1_grouped,features1_grouped),dim=1))
        features1=torch.max(features1,dim=3)[0]
        points.append(xyz1)
        out.append(features1)
        #----------------------
        xyz2=FPS(xyz1,self.k2)
        xyz2_grouped,features2_grouped=group_with_feature(xyz1,features1,xyz2,self.s2)
        features2=self.conv2(torch.cat((xyz2_grouped,features2_grouped),dim=1))
        features2=torch.max(features2,dim=3)[0]
        points.append(xyz2)
        out.append(features2)
        #------------------------

        features3=self.conv3(features2)
        global_features=torch.max(features3,dim=2)[0]
        
        return points,out,global_features






if __name__=='__main__':
    net=PCUpSample(3,[64,64]).cuda()
    a=torch.rand((3,))