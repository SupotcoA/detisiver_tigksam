import torch
from utils import Logger, check_ae, pca_weight,\
    conditional_generation, conditional_generation_gradually
import os


def train(model: torch.nn.Module,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          test_dataset):
    logger = Logger(init_val=0,
                    log_path=train_config['log_path'],
                    log_every_n_steps=train_config['log_every_n_steps'])
    for [x0, cls] in train_dataset:
        check_ae(model, x0.to(model.device), train_config['outcome_root'])
        break
    model.train()
    for [x0, cls] in train_dataset:
        x0, cls = x0.to(model.device), cls.to(model.device)
        loss, log_perplexity = model.train_step(x0, cls)
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.update(loss.detach().cpu(), log_perplexity.detach().cpu())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            test(model,
                 train_config,
                 test_dataset)
            logger.start_generation()
            model.eval()
            for cls in torch.randperm(22)[:3]:
                cls = cls.item() if not isinstance(cls, int) else cls
                conditional_generation(model, cls=cls, step=logger.step,
                                       root=train_config['outcome_root'])
            for cls in torch.randperm(22)[:3]:
                cls = cls.item() if not isinstance(cls, int) else cls
                conditional_generation_gradually(model, cls=cls, step=logger.step,
                                                 root=train_config['outcome_root'])

            logger.end_generation()
            model.train()
        if logger.step % train_config['train_steps'] == 0:
            pca_weight(weight=model.maskgit.pos_embed,
                       latent_size=None,
                       root=train_config['outcome_root'])
            if train_config['save']:
                torch.save(model.maskgit.cpu().state_dict(),
                           os.path.join(train_config['outcome_root'], f"maskgit{logger.step}.pth"))
            break


@torch.no_grad()
def test(model,
         train_config,
         test_dataset):
    model.eval()
    acc_loss = 0
    acc_log_perplexity = torch.tensor(0.0)
    step = 0
    for [x0, cls] in test_dataset:
        step += 1
        loss, log_perplexity = model.train_step(x0.to(model.device), cls.to(model.device))
        acc_loss += loss.cpu().item()
        acc_log_perplexity += log_perplexity.cpu()
    perplexity = (torch.exp(acc_log_perplexity / step)).item()
    info = f"Test step\n" \
           + f"loss: {acc_loss / step:.4f}\n" \
           + f"perplexity: {perplexity:.1f}\n"
    print(info)
    with open(train_config['log_path'], 'a') as f:
        f.write(info)
