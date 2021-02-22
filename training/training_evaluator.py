import os
import matplotlib.pyplot as plt


class TrainingEvaluator:

    def __init__(self, dir_train):
        self.dir_train = dir_train
        self.loss_burn_in = 10
        self.fid_burn_in = 0
        self.d_loss_total = 'd_loss_total'
        self.d_loss_real = 'd_loss_real'
        self.d_loss_fake = 'd_loss_fake'
        self.g_loss_total = 'g_loss_total'
        self.d_gp_loss = 'd_gp_loss'
        self.fid = 'FID'
        self.time = 'time'
        self.memory = 'memory'
        self.ada_target = 'd_ada_target'
        self.p_augment = 'p_augment'

    def plot_training_progress(self):
        loss_dict = self._read_txt_file(self.dir_train, 'loss.txt')
        fid_dict = self._read_txt_file(self.dir_train, 'fid.txt')
        plt.suptitle("Training Evaluation")
        plt.subplot(2, 3, 1)
        d_loss_real = loss_dict[self.d_loss_real][self.loss_burn_in:]
        d_loss_fake = loss_dict[self.d_loss_fake][self.loss_burn_in:]
        d_loss_total = loss_dict[self.d_loss_total][self.loss_burn_in:]
        plt.title("Discriminator Loss")
        plt.plot(d_loss_real, label='discriminator loss (real)', color='orange', linewidth=0.25)
        plt.plot(d_loss_fake, label='discriminator loss (fake)', color='blue', linewidth=0.25)
        plt.plot(d_loss_total, label='discriminator loss (total)', color='black', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 3, 2)
        g_loss = loss_dict[self.g_loss_total][self.loss_burn_in:]
        plt.title("Generator Loss")
        plt.plot(g_loss, label='generator loss', color='red', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 3, 3)
        gp_loss = loss_dict[self.d_gp_loss][self.loss_burn_in:]
        plt.title("Gradient Penalty")
        plt.plot(gp_loss, label='gradient penalty', color='purple', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 3, 4)
        plt.title("Batch Processing Time")
        time_loss = loss_dict[self.time][self.loss_burn_in:]
        plt.plot(time_loss, color='black', linewidth=0.25)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 3, 5)
        plt.title("Adaptive Image Augmentation")
        rt = loss_dict[self.ada_target][self.loss_burn_in:]
        p_augment = loss_dict[self.p_augment][self.loss_burn_in:]
        plt.plot(rt, color='blue', label='rt', linewidth=0.50)
        plt.plot(p_augment, color='orange', label='p_augment', linewidth=0.75)
        plt.grid()
        plt.tight_layout()
        plt.subplot(2, 3, 6)
        fid_loss = fid_dict[self.fid][self.fid_burn_in:]
        plt.title("Fast FID")
        plt.plot(fid_loss, label='Frechet Inception Distance', color='black', linewidth=0.25)
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

    @staticmethod
    def _str_to_float(string):
        try:
            return float(string)
        except ValueError:
            return string

    @staticmethod
    def _lod_to_dol(lod):
        return {k: [dic[k] for dic in lod] for k in lod[0]}


if __name__ == "__main__":
    name = "bob_ross_128x128"
    dir_train = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'output', name)
    evaluator = TrainingEvaluator(dir_train)
    evaluator.plot_training_progress()
