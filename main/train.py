import torch
import torch.nn as nn
from utils import load_audio, collate,log_mel_wrapper, fix_length, plot_feature
from dataset import LibriSpeech
from model import Generator, Discriminator, gen_loss, disc_loss
from torch.cuda.amp import autocast
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
import lazy_dataset
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = Experiment('Factorized_GAN', save_git_info=False)
sw = SummaryWriter()

@ex.config
def defaults():
    batch_size=16
    d_lr = 0.00005
    g_lr = 0.0001
    use_fp16 = True
    num_epochs = 500
    steps_per_eval = 1000
    L1_weight = 45
    load_ckpt = True


def prepare_dataset(dataset, batch_size):
    if isinstance(dataset, list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_audio)
    dataset = dataset.map(fix_length)
    dataset = dataset.map(log_mel_wrapper)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size, drop_last=True)
    dataset = dataset.map(collate)
 
    return dataset



@ex.automain
def main(g_lr, d_lr, use_fp16, batch_size, num_epochs, steps_per_eval, L1_weight,load_ckpt):
    steps = 0
    db = LibriSpeech()
    aux_criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    train_ds = db.get_dataset('train_clean_360')
    valid_ds = db.get_dataset('dev_clean')
    train_ds = prepare_dataset(train_ds[:], batch_size)
    valid_ds = prepare_dataset(valid_ds[:],1)

    running_loss = 0 
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    g_optim = torch.optim.AdamW(gen.parameters(),lr=g_lr)
    d_optim = torch.optim.AdamW(disc.parameters(),lr=d_lr)

    if load_ckpt:
        states = torch.load('ckpt_latest.pth')
        gen.load_state_dict(states['generator'])
        disc.load_state_dict(states['discriminator'])
        g_optim.load_state_dict(states['generator_optimizer'])
        d_optim.load_state_dict(states['discriminator_optimizer'])
        steps = states['steps']
        print(f'loaded latest checkpoint saved at {steps} steps')
    min_loss = 1e6

    for epoch in tqdm(range(num_epochs)):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_fm_loss = 0
        epoch_L1_loss = 0

        for _,batch in enumerate(tqdm(train_ds[:])):
            gen.train()
            disc.train()
            with autocast(enabled=use_fp16):
                mels = torch.stack(batch['mels']).to(device)
                
                # generator step
                fakes = gen(mels)
                d_fake, g_fm = disc(fakes)
                _, d_fm = disc(mels)
                L1_loss = aux_criterion(fakes,mels)

                fm_loss = 0
                for g,d in zip(g_fm,d_fm):
                    fm_loss += torch.mean(torch.abs(g - d))
                    
                loss = gen_loss(d_fake) + fm_loss + L1_weight * L1_loss
                scaler.scale(loss).backward()
                scaler.step(g_optim)
                scaler.update()


                epoch_g_loss += loss.item()
                epoch_fm_loss += fm_loss.item()
                epoch_L1_loss += L1_loss.item()
                running_loss += L1_loss.item()
                g_optim.zero_grad()


                # discriminator step
                d_fake ,_ = disc(fakes.detach())
                d_real,_ = disc(mels)
                loss_d = disc_loss(d_real, d_fake) 
                scaler.scale(loss_d).backward()
                scaler.step(d_optim)
                scaler.update()
                d_optim.zero_grad()
                epoch_d_loss += loss_d.item()
   

            if steps % steps_per_eval == 0:
                eval_g_loss = 0

                print(f'running L1_loss at {steps}: {running_loss / (steps+1)}.')
                gen.eval()
                disc.eval()
                for _ ,batch in enumerate(tqdm(valid_ds[:])):
                    mels = torch.stack(batch['mels']).to(device)
                    with autocast(enabled=False):
                        with torch.no_grad():
                            fakes = gen(mels)
                            d_fake, _ = disc(fakes)
                            d_real, _ = disc(mels)
                            g_loss = aux_criterion(fakes,mels)
                            eval_g_loss += g_loss.item()
                gen.train()
                disc.train()
                print('model set back to train_mode')
                sw.add_scalar('validation/epoch_loss', eval_g_loss/len(valid_ds), steps)
                sw.add_figure('validation/original_spectrogram', plot_feature(mels.squeeze().cpu()),steps)
                sw.add_figure('validation/reconstructed_spectrogram', plot_feature(fakes.squeeze().cpu()), steps)
                print(f'validation L1 loss: {eval_g_loss / len(valid_ds)}')
                if eval_g_loss/ len(valid_ds) < min_loss:
                    min_loss = eval_g_loss / len(valid_ds)
                    torch.save({
                                'steps':steps,
                                'generator': gen.state_dict(),
                                'generator_optimizer': g_optim.state_dict(),
                                'discriminator': disc.state_dict(),
                                'discriminator_optimizer':d_optim.state_dict()}, 'ckpt_best_loss.pth')
                torch.save({
                            'steps': steps,
                            'generator': gen.state_dict(),
                            'generator_optimizer': g_optim.state_dict(),
                            'discriminator': disc.state_dict(),
                            'discriminator_optimizer': d_optim.state_dict(),
                            }, 'ckpt_latest.pth')
            steps +=1     

        print(f'epoch {epoch + 1} loss: {epoch_g_loss / len(train_ds)}')
        print(f'epoch {epoch + 1} loss: {epoch_d_loss / len(train_ds)}')

        sw.add_scalar('training/generator_loss', epoch_g_loss / len(train_ds), epoch)
        sw.add_scalar('training/feature_matching_loss', epoch_fm_loss / len(train_ds), epoch)
        sw.add_scalar('training/L1_loss', epoch_L1_loss / len(train_ds), epoch)
        sw.add_scalar('training/discriminator_loss', epoch_d_loss / len(train_ds), epoch)    



