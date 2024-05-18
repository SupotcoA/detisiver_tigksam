import numpy as np
import cv2
import os
import torch
import time
import matplotlib.pyplot as plt


@torch.no_grad()
def print_num_params(model, name, log_path):
    num_params = 0
    if isinstance(model, torch.nn.Module):
        for param in model.parameters():
            num_params += param.numel()
    elif isinstance(model, list):
        for param in model:
            num_params += param.numel()
    with open(log_path, 'a') as f:
        f.write(f"{name} parameters: {num_params}\n")
    print(f"{name} parameters: {num_params}")


@torch.no_grad()
def tensor2bgr(tensor):
    imgs = torch.clip(torch.permute(tensor, [0, 2, 3, 1]).cpu().add(1).mul(127.5), 0, 255)
    return imgs.numpy().astype(np.uint8)[:, :, :, ::-1]


@torch.no_grad()
def bgr2tensor(bgr):
    imgs = torch.from_numpy(bgr[:, :, ::-1]) / 127.5 - 1
    return torch.permute(imgs, [0, 3, 1, 2])


@torch.no_grad()
def vis_imgs(imgs, step, cls, root, use_plt=False):
    if not isinstance(imgs, np.ndarray):
        imgs = tensor2bgr(imgs)
    if imgs.shape[0] > 9:
        imgs = imgs[:9]
    elif imgs.shape[0] < 9:
        raise ValueError(f"{imgs.shape}")
    h, w, c = imgs.shape[1:]
    base = np.zeros((h * 3, w * 3, c), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            base[i * h:i * h + h, j * w:j * w + w, :] = imgs[i * 3 + j]
    fp = os.path.join(root, f"cd{step}_{cls}.png")
    cv2.imwrite(fp, base)
    if use_plt:
        plt.imshow(base[:, :, ::-1])
        plt.show()


@torch.no_grad()
def vis_imgs_gradually(imgs, step, cls, root, use_plt=False):
    n_steps, b, h, w, c = imgs.shape
    base = np.zeros((b * h, n_steps * w, c), dtype=np.uint8)
    for i in range(b):
        for j in range(n_steps):
            base[i * h:i * h + h, j * w:j * w + w, :] = imgs[j, i]
    fp = os.path.join(root, f"cdp{step}_{cls}.png")
    cv2.imwrite(fp, base)
    if use_plt:
        plt.imshow(base[:, :, ::-1])
        plt.show()


class Logger:
    def __init__(self,
                 init_val=0,
                 log_path=None,
                 log_every_n_steps=None):
        self.loss = 0
        self.log_perplexity = 0
        self.step = 0
        self.log_path = log_path
        self.log_every_n_steps = log_every_n_steps
        self.time = 0
        self.eval_time = 0

    def update(self, loss, log_perplexity):
        if self.loss == 0:
            self.time = time.time()
        self.loss += loss
        self.log_perplexity += log_perplexity
        self.step += 1
        if self.step % self.log_every_n_steps == 0:
            self.log()
            self.loss = 0
            self.log_perplexity = 0

    def log(self):
        if not isinstance(self.log_perplexity, int):
            self.log_perplexity = self.log_perplexity.item()
        perplexity = np.exp(self.log_perplexity / self.log_every_n_steps)
        dt = time.time() - self.time
        info = f"Train step {self.step}\n" \
               + f"loss: {self.loss / self.log_every_n_steps:.4f}\n" \
               + f"perplexity: {perplexity:.1f}\n" \
               + f"time: {dt:.1f} \n"
        print(info)
        with open(self.log_path, 'a') as f:
            f.write(info)

    def start_generation(self):
        self.eval_time = time.time()

    def end_generation(self):
        dt = time.time() - self.eval_time
        info = f"generation time: {dt:.2f}\n"
        print(info)
        with open(self.log_path, 'a') as f:
            f.write(info)


@torch.no_grad()
def check_ae(model, x, root):
    if x.shape[0] < 9:
        return
    imgs = model.decode(x[:9])
    vis_imgs(imgs, "ae_check", "ae_check", root, use_plt=True)


@torch.no_grad()
def conditional_generation(model, cls: int, step, root):
    idx = model.conditional_generation(temperature=(2, 1), n_steps=9, b=9, c=cls)  # [b, n_pos]
    imgs = model.decode(idx)
    vis_imgs(imgs, step, cls, root, use_plt=False)


@torch.no_grad()
def conditional_generation_gradually(model, cls: int, step, root):
    b = 4
    n_steps = 9
    idx = model.conditional_generation_gradually(temperature=(1, 1), n_steps=n_steps, b=b, c=cls)  # [b, n_pos]
    imgs = []
    for i in range(n_steps):
        imgs.append(tensor2bgr(model.decode(idx[i])))  # imgs [n_steps, b, h, w, c=3]
    vis_imgs_gradually(np.asarray(imgs), step, cls, root, use_plt=False)
