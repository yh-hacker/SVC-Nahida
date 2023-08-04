import logging
import multiprocessing
import time

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import paddle
#paddle.device.set_device("cpu") #开启可用CPU进行炼丹
trainer:str = "admin" 
from paddle.nn import functional as F
from paddle.io import DataLoader
from visualdl import LogWriter
from paddle.amp import auto_cast, GradScaler

import modules.commons as commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioCollate
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from modules.losses import (
    kl_loss,
    generator_loss, discriminator_loss, feature_loss
)

from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

paddle.set_flags({'FLAGS_cudnn_exhaustive_search': True}) # 使用穷举搜索方法来选择卷积算法
global_step = 0
trainers:list[str] = []
start_time = time.time()


def main():
    """Assume Single Node Multi GPUs Training Only"""
    #assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = paddle.device.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    run(n_gpus, hps, )


def run(n_gpus, hps):
    global global_step,trainers,trainer

    trainer = hps.trainer
    rank = 0
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = LogWriter(logdir=hps.model_dir)
        writer_eval = LogWriter(logdir=os.path.join(hps.model_dir, "eval"))

    paddle.seed(hps.train.seed)
    paddle.device.set_device('cpu' if paddle.device.get_device() == 'cpu' else 'gpu:' + str(rank))
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    train_loader = DataLoader(dataset = train_dataset,
                             num_workers=num_workers,
                             shuffle=False,
                             batch_size=hps.train.batch_size,
                             collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(dataset = eval_dataset,
                                num_workers = 1,
                                shuffle = False,
                                batch_size = 1,
                                drop_last = False,
                                collate_fn = collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    optim_g = paddle.optimizer.AdamW(
        parameters = net_g.parameters(),
        learning_rate = hps.train.learning_rate,
        beta1 = hps.train.betas[0],
        beta2 = hps.train.betas[1],
        epsilon = hps.train.eps)
    optim_d = paddle.optimizer.AdamW(
        parameters = net_d.parameters(),
        learning_rate = hps.train.learning_rate,
        beta1 = hps.train.betas[0],
        beta2 = hps.train.betas[1],
        epsilon = hps.train.eps)

    skip_optimizer = False
    try:
        _, _, _, epoch_str, trainers = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pdparams"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str, trainers = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pdparams"), net_d,
                                                   optim_d, skip_optimizer)
        if trainer not in trainers:
            trainers.append(trainer)
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        print(e)
        logger.info("加载旧检查点失败……")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = paddle.optimizer.lr.ExponentialDecay(hps.train.learning_rate, gamma = hps.train.lr_decay, last_epoch = epoch_str - 2)
    scheduler_d = paddle.optimizer.lr.ExponentialDecay(hps.train.learning_rate, gamma = hps.train.lr_decay, last_epoch = epoch_str - 2)

    optim_g = paddle.optimizer.AdamW(
        parameters = net_g.parameters(),
        learning_rate = scheduler_g,
        beta1 = hps.train.betas[0],
        beta2 = hps.train.betas[1],
        epsilon = hps.train.eps)
    optim_d = paddle.optimizer.AdamW(
        parameters = net_d.parameters(),
        learning_rate = scheduler_d,
        beta1 = hps.train.betas[0],
        beta2 = hps.train.betas[1],
        epsilon = hps.train.eps)

    scaler = GradScaler(enable = hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler:GradScaler, loaders, logger:logging.Logger, writers:list or None):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv = items
        g = spk
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        with auto_cast(enable=hps.train.fp16_run):

            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with auto_cast(enable=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.clear_grad()
        scaler.scale(loss_disc_all).backward(retain_graph = True) # 将 Tensor 乘上缩放因子，返回缩放后的输出，返回loss然后反向传播
        scaler.unscale_(optim_d) # 将参数的梯度除去缩放比例。
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        with auto_cast(enable=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with auto_cast(enable=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
        optim_g.clear_grad()
        scaler.scale(loss_gen_all).backward(retain_graph = True)
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        #lr = optim_g.state_dict()['LR_Scheduler']['last_lr'] # paddle优化器特有的字典
        #losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        #logger.info(f"损失：{[x.item() for x in losses]}，步数：{global_step}，学习率：{lr}") # 梅花自己看的~

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.state_dict()['LR_Scheduler']['last_lr'] # paddle优化器特有的字典
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('训练回合：{} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"损失：{[x.item() for x in losses]}，步数：{global_step}，学习率：{lr}")

                scalar_dict = {"损失/生成器/总损失": loss_gen_all, "损失/判别器/总损失": loss_disc_all, "学习率": lr,
                               "归一化判别器梯度": grad_norm_d, "归一化生成器梯度": grad_norm_g}
                scalar_dict.update({"损失/生成器/特征匹配损失": loss_fm, "损失/生成器/梅尔频谱损失": loss_mel, "损失/生成器/KL散度": loss_kl,
                                    "损失/生成器/基音损失": loss_lf0})

                image_dict = {
                    "切片/原始梅尔频谱图": utils.plot_spectrogram_to_numpy(y_mel[0].detach().numpy()),
                    "切片/生成梅尔频谱图": utils.plot_spectrogram_to_numpy(y_hat_mel[0].detach().numpy()),
                    "全部/梅尔频谱图": utils.plot_spectrogram_to_numpy(mel[0].detach().numpy()),
                    "全部/基音损失": utils.plot_data_to_numpy(lf0[0, 0, :].numpy(),
                                                          pred_lf0[0, 0, :].detach().numpy()),
                    "全部/归一化基音损失": utils.plot_data_to_numpy(lf0[0, 0, :].numpy(),
                                                               norm_lf0[0, 0, :].detach().numpy())
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )
            if global_step % hps.train.eval_interval == 0:
                if hps.clean_logs:
                    os.system('clear')
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pdparams".format(global_step)), trainers)
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pdparams".format(global_step)), trainers)
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> 回合：{epoch}, 消耗 {durtaion} 秒')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with paddle.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv = items
            g = spk[:1]
            spec, y = spec[:1], y[:1]
            c = c[:1]
            f0 = f0[:1]
            uv= uv[:1]
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat = generator.infer(c, f0, uv, g=g)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).cast('float32'),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"生成器测试数据/音频_{batch_idx}": y_hat[0],
                f"地标真实数据/音频_{batch_idx}": y[0]
            })
        image_dict.update({
            "生成器测试数据/梅尔频谱图": utils.plot_spectrogram_to_numpy(y_hat_mel[0].numpy()),
            "地标真实数据/梅尔频谱图": utils.plot_spectrogram_to_numpy(mel[0].numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
