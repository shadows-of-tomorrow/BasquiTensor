import os
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt


class TrainingEvaluator:

    def __init__(self, image_processor=None, loss_file_name='loss.txt'):
        self.image_processor = image_processor
        self.loss_file_name = loss_file_name
        self.d_loss_real = ' d_loss_real'
        self.d_loss_fake = ' d_loss_fake'
        self.g_loss = ' g_loss'
        self.gp_loss = ' gp_loss'

    def _compute_fid_dict(self):
        networks_dir = os.path.join(self.image_processor.dir_out, 'networks')
        for network_dir in os.listdir(networks_dir):
            resolution = int(network_dir.split('_')[0][0])
            fade_in = network_dir.split('_')[1] == "fade_in"
            generator_path = os.path.join(networks_dir, network_dir, 'generator')

    def _compute_fid(self, model, x_real, x_fake):
        # 1. Compute activations.
        activation_real = model.predict(x_real)
        activation_fake = model.predict(x_fake)
        # 2. Compute mean and covariance.
        mu_real, sigma_real = activation_real.mean(axis=0), np.cov(activation_real, rowvar=False)
        mu_fake, sigma_fake = activation_fake.mean(axis=0), np.cov(activation_fake, rowvar=False)
        # 3. Compute difference statistics.
        mu_diff = np.sum(np.square(mu_real-mu_fake))
        cov_diff = scp.linalg.sqrtm(sigma_real.dot(sigma_fake))
        # 4. Check if imaginary numbers from matrix square root.
        if np.iscomplexobj(cov_diff):
            cov_diff = cov_diff.real
        # 5. Compute Frechet Inception Distance (FID).
        return mu_diff + np.trace(sigma_real + sigma_fake - 2.0 * cov_diff)

    def _plot_loss(self, loss_dict):
        plt.suptitle("Training Loss")
        plt.subplot(2, 2, 1)
        d_loss = self._add_d_losses(loss_dict)
        plt.plot(d_loss, label='discriminator loss (total)', color='black', linewidth=0.25)
        plt.tight_layout()
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(loss_dict[self.g_loss], label='generator loss', color='red', linewidth=0.25)
        plt.tight_layout()
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(loss_dict[self.gp_loss], label='gradient penalty', color='purple', linewidth=0.25)
        plt.tight_layout()
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(loss_dict[self.d_loss_real], label='discriminator loss (real)', color='orange', linewidth=0.25)
        plt.plot(loss_dict[self.d_loss_fake], label='discriminator loss (fake)', color='blue', linewidth=0.25)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def _parse_loss_file(self, loss_dir):
        with open(loss_dir + '/' + self.loss_file_name, 'r') as file:
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