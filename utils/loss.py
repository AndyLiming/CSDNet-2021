import torch
import torch.nn as nn
import torch.nn.functional as F


def multiScaleLoss(input, target, maxDepth):
  assert (type(input) is tuple and len(input) == 6)
  loss = torch.Tensor([0.0]).float().cuda()
  scale = 1
  weight = 1 / 2
  for i in range(0, 6):
    gt = F.interpolate(target, scale_factor=scale)
    mask = (gt <= maxDepth).float()
    loss += weight * F.mse_loss(input[i] * mask, gt * mask)
    scale /= 2
    weight /= 2
    if weight < 1 / 32: weight = 1 / 32

  # loss += 1 / 4 * F.mse_loss(input[1], F.interpolate(target, scale_factor=1 / 2))
  # loss += 1 / 8 * F.mse_loss(input[2], F.interpolate(target, scale_factor=1 / 4))
  # loss += 1 / 16 * F.mse_loss(input[3], F.interpolate(target, scale_factor=1 / 8))
  # loss += 1 / 32 * F.mse_loss(input[4], F.interpolate(target, scale_factor=1 / 16))
  # loss += 1 / 32 * F.mse_loss(input[5], F.interpolate(target, scale_factor=1 / 32))
  return loss


def multiScaleWeightedLoss(input, target, maxDepth, levels):
  assert (type(input) is tuple and len(input) == levels)
  loss = torch.Tensor([0.0]).float().cuda()
  scale = 1
  weight = 1 / 2
  minW = (1 / 2)**levels
  for i in range(0, levels):
    gt = F.interpolate(target, scale_factor=scale)
    mask = (gt <= maxDepth).float()
    loss += weight * F.mse_loss(input[i] * mask, gt * mask)
    scale /= 2
    weight /= 2
    if weight < minW: weight = minW
  return loss


def depLoss(pred, gt, maxDepth):
  mask = (gt < maxDepth).float()
  loss = F.mse_loss(pred * mask, gt * mask)
  return loss


def berhuLossS2(pred, gt, mask):
  diff = gt - pred
  abs_diff = torch.abs(diff)
  c = torch.max(abs_diff).item() / 5
  leq = (abs_diff <= c).float()
  l2_losses = (diff**2 + c**2) / (2 * c)
  loss = leq * abs_diff + (1 - leq) * l2_losses
  count = torch.sum(mask, dim=[1, 2], keepdim=True).float()
  masked_loss = loss * mask.float()
  return torch.mean(torch.sum(masked_loss, dim=[1, 2], keepdim=True) / count)


def berhuLoss(pred, gt, mask, weights):
  diff = gt - pred
  abs_diff = torch.abs(diff)
  c = torch.max(abs_diff).item() / 5
  leq = (abs_diff <= c).float()
  l2_losses = (diff**2 + c**2) / (2 * c)
  loss = leq * abs_diff + (1 - leq) * l2_losses
  count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
  masked_loss = loss * mask.float()
  weighted_loss = masked_loss * weights
  return torch.mean(torch.sum(weighted_loss, dim=[1, 2, 3], keepdim=True) / count)


def guided_smoothness_loss(input_duv, guide_duv, mask):
  guidance_weights = torch.exp(-guide_duv)
  smoothness = input_duv * guidance_weights
  smoothness[~mask] = 0.0
  return torch.sum(smoothness) / torch.sum(mask)