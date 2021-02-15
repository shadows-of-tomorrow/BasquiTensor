import numpy as np
import matplotlib.pyplot as plt


class TrainingEvaluator:

    def __init__(self, image_processor=None):
        self.loss_burn_in = 0
        self.fid_burn_in = 0
        self.image_processor = image_processor
        self.d_loss_real = 'd_loss_real'
        self.d_loss_fake = 'd_loss_fake'
        self.g_loss = 'g_loss'
        self.gp_loss = 'gp_loss'
        self.fid = 'FID'

    def plot_loss(self, loss_dir):
        loss_dict = self._read_txt_file(loss_dir, 'loss.txt')
        fid_dict = self._read_txt_file(loss_dir, 'fid.txt')
        plt.suptitle("Training Evaluation")
        plt.subplot(2, 2, 1)
        d_loss_real = self._scale_losses(loss_dict[self.d_loss_real][self.loss_burn_in:])
        d_loss_fake = self._scale_losses(loss_dict[self.d_loss_fake][self.loss_burn_in:])
        d_loss = self._scale_losses(self._add_d_losses(loss_dict)[self.loss_burn_in:])
        plt.title("Discriminator Loss")
        plt.plot(d_loss_real, label='discriminator loss (real)', color='orange', linewidth=0.25)
        plt.plot(d_loss_fake, label='discriminator loss (fake)', color='blue', linewidth=0.25)
        plt.plot(d_loss, label='discriminator loss (total)', color='black', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 2, 2)
        g_loss = self._scale_losses(loss_dict[self.g_loss][self.loss_burn_in:])
        plt.title("Generator Loss")
        plt.plot(g_loss, label='generator loss', color='red', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 2, 3)
        gp_loss = self._scale_losses(loss_dict[self.gp_loss][self.loss_burn_in:])
        plt.title("Gradient Penalty")
        plt.plot(gp_loss, label='gradient penalty', color='purple', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 2, 4)
        fid_loss = fid_dict[self.fid][self.fid_burn_in:]
        plt.title("Frechet Inception Distance")
        plt.plot(fid_loss, label='Frechet Inception Distance', color='green', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.show()

    def _read_txt_file(self, loss_dir, name):
        with open(loss_dir + '/' + name, 'r') as file:
            lines = file.readlines()
            losses = []
        for line in lines:
            line = self._parse_line(line)
            losses.append(line)
        losses = self._lod_to_dol(losses)
        return losses

    def _parse_line(self, line):
        line = line.split(",")[:-1]
        line = dict([item.split(':') for item in line])
        for key, value in line.items():
            value = self._str_to_float(value)
            line[key] = value
        return line

    def _add_d_losses(self, loss_dict):
        return [loss_dict[self.d_loss_real][k]
                + loss_dict[self.d_loss_fake][k]
                for k in range(len(loss_dict[self.d_loss_fake]))]

    @staticmethod
    def _str_to_float(string):
        try:
            return float(string)
        except ValueError:
            return string

    @staticmethod
    def _lod_to_dol(lod):
        return {k: [dic[k] for dic in lod] for k in lod[0]}

    @staticmethod
    def _scale_losses(losses):
        return [np.sqrt(x) if x > 0 else -np.sqrt(-x) for x in losses]

    @staticmethod
    def _apply_ewma(losses, alpha=0.99):
        for k in range(1, len(losses)):
            losses[k] = alpha * losses[k-1] + (1.0-alpha) * losses[k]
        return losses


dir_loss = f"C:\\Users\\robin\\Desktop\\Projects\\painter\\io\\output\\celeb_a_256x256"
TrainingEvaluator().plot_loss(dir_loss)
