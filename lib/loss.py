import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

from lib.utils import (
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image
)

from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')


def loss_function(
        model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False
):
    output = model({
        'image1': batch['image1'].to(device),
        'image2': batch['image2'].to(device)
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):
        # Annotations
        depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]

        depth2 = batch['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch['bbox2'][idx_in_batch].to(device)

        # Network output
        dense_features1 = output['dense_features1'][idx_in_batch]
        c, h1, w1 = dense_features1.size()
        scores1 = output['scores1'][idx_in_batch].view(-1)

        dense_features2 = output['dense_features2'][idx_in_batch]
        _, h2, w2 = dense_features2.size()
        scores2 = output['scores2'][idx_in_batch]

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        descriptors1 = all_descriptors1

        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

        # Warp the positions from image 1 to image 2
        fmap_pos1 = grid_positions(h1, w1, device)
        pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps)
        try:
            pos1, pos2, ids = warp(
                pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2
            )
        except EmptyTensorError:
            continue
        fmap_pos1 = fmap_pos1[:, ids]
        descriptors1 = descriptors1[:, ids]
        scores1 = scores1[ids]

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            continue

        # Descriptors at the corresponding positions
        fmap_pos2 = torch.round(
            downscale_positions(pos2, scaling_steps=scaling_steps)
        ).long()
        descriptors2 = F.normalize(
            dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
            dim=0
        )
        positive_distance = 2 - 2 * (
            descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
        ).squeeze()

        all_fmap_pos2 = grid_positions(h2, w2, device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos2.unsqueeze(2).float() -
                all_fmap_pos2.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]

        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(
            torch.abs(
                fmap_pos1.unsqueeze(2).float() -
                all_fmap_pos1.unsqueeze(1)
            ),
            dim=0
        )[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1
        )[0]

        diff = positive_distance - torch.min(
            negative_distance1, negative_distance2
        )

        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

        #loss = loss + (
        #    torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
        #    torch.sum(scores1 * scores2)
        #)
        cosim = torch.nn.CosineSimilarity(dim=0)
        classes1 = []
        classes2 = []
        for i in range(ids.size()[0]):
            classes1.append(i)
            classes2.append(i)
        classes1 = torch.tensor(classes1)
        classes2 = torch.tensor(classes2)
        classes = torch.cat([classes1, classes2])
        fast_ap_loss = FastAPLoss()
        loss = loss + (0.3*(1 - cosim(scores1, scores2)) + 0.1*(1 - torch.sum(scores1)) + 0.1*(1 - torch.sum(scores2)) + 
               0.5*fast_ap_loss(torch.cat([descriptors1, descriptors2], dim=1).t(), classes))

        has_grad = True
        n_valid_samples += 1

        if plot and batch['batch_idx'] % batch['log_interval'] == 0:
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure()
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                batch['image1'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :], pos1_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 2)
            plt.imshow(
                output['scores1'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                batch['image2'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :], pos2_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            plt.subplot(1, n_sp, 4)
            plt.imshow(
                output['scores2'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            savefig('train_vis/%s.%02d.%02d.%d.png' % (
                'train' if batch['train'] else 'valid',
                batch['epoch_idx'],
                batch['batch_idx'] // batch['log_interval'],
                idx_in_batch
            ), dpi=300)
            plt.close()

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def warp(
        pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2
):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention
    u1 = pos1[1, :] + bbox1[1] + .5
    v1 = pos1[0, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)
    XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids

def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D-mid)/Delta
    return torch.max(torch.Tensor([0]).cuda(), y)

def dSoftBinning(D, mid, Delta):
    side1 = (D > (mid - Delta)).type(torch.float)
    side2 = (D <= mid).type(torch.float)
    ind1 = (side1 * side2) #.type(torch.uint8)

    side1 = (D > mid).type(torch.float)
    side2 = (D <= (mid + Delta)).type(torch.float)
    ind2 = (side1 * side2) #.type(torch.uint8)

    return (ind1 - ind2)/Delta


class FastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition
    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size donesn't match!"

        # 1. get affinity matrix
        Y   = target.unsqueeze(1)
        Aff = 2 * (Y == Y.t()).type(torch.float) - 1
        Aff.masked_fill_(torch.eye(N, N).bool(), 0)  # set diagonal to 0

        I_pos = (Aff > 0).type(torch.float).cuda()
        I_neg = (Aff < 0).type(torch.float).cuda()
        N_pos = torch.sum(I_pos, 1)

        # 2. compute distances from embeddings
        # squared Euclidean distance with range [0,4]
        dist2 = 2 - 2 * torch.mm(input, input.t())

        # 3. estimate discrete histograms
        Delta = torch.tensor(4. / num_bins).cuda()
        Z     = torch.linspace(0., 4., steps=num_bins+1).cuda()
        L     = Z.size()[0]
        h_pos = torch.zeros((N, L)).cuda()
        h_neg = torch.zeros((N, L)).cuda()
        for l in range(L):
            pulse      = softBinning(dist2, Z[l], Delta)
            h_pos[:,l] = torch.sum(pulse * I_pos, 1)
            h_neg[:,l] = torch.sum(pulse * I_neg, 1)

        H_pos = torch.cumsum(h_pos, 1)
        h     = h_pos + h_neg
        H     = torch.cumsum(h, 1)

        # 4. compate FastAP
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP, 1) / N_pos
        FastAP = FastAP[ ~torch.isnan(FastAP) ]

        loss   = 1 - torch.mean(FastAP)

        # 5. save for backward
        ctx.save_for_backward(input, target)
        ctx.Z     = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h     = h
        ctx.H     = H
        ctx.L     = torch.tensor(L)

        return loss


    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        Z     = Variable(ctx.Z     , requires_grad = False)
        Delta = Variable(ctx.Delta , requires_grad = False)
        dist2 = Variable(ctx.dist2 , requires_grad = False)
        I_pos = Variable(ctx.I_pos , requires_grad = False)
        I_neg = Variable(ctx.I_neg , requires_grad = False)
        h     = Variable(ctx.h     , requires_grad = False)
        H     = Variable(ctx.H     , requires_grad = False)
        h_pos = Variable(ctx.h_pos , requires_grad = False)
        h_neg = Variable(ctx.h_neg , requires_grad = False)
        H_pos = Variable(ctx.H_pos , requires_grad = False)
        N_pos = Variable(ctx.N_pos , requires_grad = False)

        L     = Z.size()[0]
        H2    = torch.pow(H,2)
        H_neg = H - H_pos

        # 1. d(FastAP)/d(h+)
        LTM1 = torch.tril(torch.ones(L,L), -1)  # lower traingular matrix
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0

        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1.cuda())
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L,1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0


        # 2. d(FastAP)/d(h-)
        LTM0 = torch.tril(torch.ones(L,L), 0)  # lower triangular matrix
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0

        d_AP_h_neg = torch.mm(tmp2, LTM0.cuda())
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L,1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0


        # 3. d(FastAP)/d(embedding)
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg

            alpha_p = torch.diag(d_AP_h_pos[:,l]) # N*N
            alpha_n = torch.diag(d_AP_h_neg[:,l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)

            # accumulate gradient
            d_AP_x = d_AP_x - torch.mm(input.t(), (Ap+An))

        grad_input = -d_AP_x
        return grad_input.t(), None, None


class FastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition
    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    NOTE:
    Given an input batch, FastAP does not sample triplets from it as it's not a
    triplet-based method. Therefore, FastAP does not take a Sampler as input.
    Rather, we need to specify how the input batch is selected, separately.
    """
    def __init__(self, num_bins=10):
        super(FastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return FastAP.apply(batch, labels, self.num_bins)