from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(
        self,
        in_ch=3,
        mlp_specs=None,
        global_feat=True,
        xyz_transform=False,
        feature_transform=False,
    ):
        super(PointNetfeat, self).__init__()
        if mlp_specs is None:
            mlp_specs = [64, 128, 512]
        assert in_ch >= 3
        self.in_ch = in_ch
        self.mlp_specs = mlp_specs
        self.global_feat = global_feat
        self.xyz_transform = xyz_transform
        if self.xyz_transform:
            self.stn = STN3d()
        # self.fc = nn.Sequential(
        #     nn.Linear(mlp_specs[2], 256),
        #     nn.Linear(256, 128)
        # )
        self.conv1 = torch.nn.Conv1d(in_ch, mlp_specs[0], 1)
        self.conv2 = torch.nn.Conv1d(mlp_specs[0], mlp_specs[1], 1)
        self.conv3 = torch.nn.Conv1d(mlp_specs[1], mlp_specs[2], 1)
        # self.bn1 = nn.BatchNorm1d(mlp_specs[0])
        # self.bn2 = nn.BatchNorm1d(mlp_specs[1])
        # self.bn3 = nn.BatchNorm1d(mlp_specs[2])
        self.norm1 = nn.LayerNorm(mlp_specs[0], eps=1e-6)
        self.norm2 = nn.LayerNorm(mlp_specs[1], eps=1e-6)
        self.norm3 = nn.LayerNorm(mlp_specs[2], eps=1e-6)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        if self.xyz_transform:
            if self.in_ch > 3:
                xyz, feat = x[:, :3], x[:, 3:]
                trans = self.stn(xyz)
                xyz = xyz.transpose(2, 1)
                xyz = torch.bmm(xyz, trans)
                xyz = xyz.transpose(2, 1)
                x = torch.cat((xyz, feat), dim=1)
            else:
                trans = self.stn(x)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans)
                x = x.transpose(2, 1)
        else:
            trans = None
            x = F.relu(self.norm1(self.conv1(x).transpose(2, 1)))
            x = x.transpose(2, 1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.norm2(self.conv2(x).transpose(2, 1)))
        x = x.transpose(2, 1)
        x = self.norm3(self.conv3(x).transpose(2, 1))
        x = x.transpose(2, 1)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.mlp_specs[2])
        if self.global_feat:
            # x = self.fc(x)
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.mlp_specs[2], 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        # global_x = self.fc(x)
        # cat_x = torch.cat([x.view(-1, self.mlp_specs[2], 1).repeat(1, 1, n_pts), pointfeat], 1)
        # return global_x, cat_x, trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(
        self, in_ch=3, mlp_specs=None, k=2, xyz_transform=False, feature_transform=False
    ):
        super(PointNetDenseCls, self).__init__()
        if mlp_specs is None:
            mlp_specs = [64, 128, 512]
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            in_ch=in_ch,
            mlp_specs=mlp_specs,
            xyz_transform=xyz_transform,
            feature_transform=feature_transform,
        )
        self.conv1 = torch.nn.Conv1d(mlp_specs[0] + mlp_specs[2], 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # batchsize = x.size()[0]
        # n_pts = x.size()[2]
        gloabal_x, cat_x, trans, trans_feat = self.feat(x)  # (N, 128), (1088, N)
        cat_x = F.relu(self.bn1(self.conv1(cat_x)))
        cat_x = F.relu(self.bn2(self.conv2(cat_x)))
        cat_x = F.relu(self.bn3(self.conv3(cat_x)))
        cat_x = self.conv4(cat_x)
        cat_x = cat_x.transpose(2, 1).contiguous()  # (N, 2)
        # cat_x = F.log_softmax(cat_x.view(-1,self.k), dim=-1)
        # cat_x = cat_x.view(batchsize, n_pts, self.k)
        return gloabal_x, cat_x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print("stn", out.size())
    print("loss", feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print("stn64d", out.size())
    print("loss", feature_transform_regularizer(out))

    pointfeat = PointNetfeat()
    global_x, cat_x, _, _ = pointfeat(sim_data)
    print("global feat, point feat: ", global_x.size(), cat_x.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    seg = PointNetDenseCls(k=2)
    global_x, cat_x, _, _ = seg(sim_data)
    print("global feat, seg map: ", global_x.size(), cat_x.size())
