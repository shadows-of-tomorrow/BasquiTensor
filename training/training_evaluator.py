import os
import matplotlib.pyplot as plt


class TrainingEvaluator:

    def __init__(self, loss_file_name='loss.txt'):
        self.loss_file_name = loss_file_name


    def plot_loss(self, loss_dict):
        plt.suptitle("Training Loss")
        plt.subplot(2, 2, 1)
        d_loss = [loss_dict[' d_loss_real'][k] + loss_dict[' d_loss_fake'][k] for k in range(len(loss_dict[' d_loss_real']))]
        plt.plot(d_loss, label='discriminator loss (total)', color='black', linewidth=0.250)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(loss_dict[' g_loss'], label='generator loss', color='red', linewidth=0.25)
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(loss_dict[' gp_loss'], label='gradient penalty', color='purple', linewidth=0.25)
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(loss_dict[' d_loss_fake'], label='discriminator loss (fake)', color='blue', linewidth=0.25)
        plt.plot(loss_dict[' d_loss_real'], label='discriminator loss (real)', color='orange', linewidth=0.25)
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

    @staticmethod
    def _str_to_float(string):
        try:
            return float(string)
        except ValueError:
            return string

    @staticmethod
    def _lod_to_dol(lod):
        return {k: [dic[k] for dic in lod] for k in lod[0]}


loss_dir = os.path.dirname(os.path.dirname(__file__))
run_folder = 'celeb_a_256x256_pggan'
loss_dir = os.path.join(loss_dir, 'io', 'output', run_folder)

eval = TrainingEvaluator()
loss_dict = eval._parse_loss_file(loss_dir)
eval.plot_loss(loss_dict)