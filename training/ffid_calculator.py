import numpy as np
from gans.utils import generate_real_samples
from gans.utils import generate_fake_samples
from tensorflow.keras.applications.inception_v3 import InceptionV3


class FFIDCalculator:

    def __init__(self, image_processor, fid_res=128, m=128, n=10000, d=2048):
        print(f"Constructing Fast Frechet Inception Distance calculator...")
        self.image_processor = image_processor
        self.m, self.n, self.d = m, n, d
        self.img_shape = (fid_res, fid_res)
        self.fid_shape = (fid_res, fid_res, 3)
        self.inception_network = InceptionV3(include_top=False, pooling='avg', input_shape=self.fid_shape)
        self.mu_real, self.c_real, self.trc_sigma_real = self._precompute_values()

    def compute_fid(self, generator):
        x_fake, _ = generate_fake_samples(generator, generator.input_shape[1], self.m)
        x_fake = self.image_processor.resize_np_array(x_fake, self.img_shape)
        a_fake = np.transpose(self.inception_network.predict(x_fake))
        mu_fake = a_fake.mean(axis=1).reshape(-1, 1)
        c_fake = self._compute_c_matrix(a_fake, mu_fake, self.m)
        trc_sqrt_mat = self._compute_trace_sqrt_mat(c_fake)
        trc_sigma_fake = self._compute_trace_cov_mat(c_fake)
        fid = np.sum(np.square(mu_fake-self.mu_real)) + trc_sigma_fake + self.trc_sigma_real - 2 * trc_sqrt_mat
        return fid

    def _precompute_values(self):
        x_real, _ = generate_real_samples(self.image_processor, self.n, self.img_shape)
        a_real = np.transpose(self.inception_network.predict(x_real))
        mu_real = a_real.mean(axis=1).reshape(-1, 1)
        c_real = self._compute_c_matrix(a_real, mu_real, self.n)
        trc_sigma_real = self._compute_trace_cov_mat(c_real)
        return mu_real, c_real, trc_sigma_real

    @staticmethod
    def _compute_c_matrix(a, mu, n_examples):
        return (1.0 / np.sqrt(n_examples-1)) * (a - np.matmul(mu, np.ones((1, n_examples))))

    @staticmethod
    def _compute_trace_cov_mat(c):
        trc_sigma = 0.0
        for i in range(c.shape[0]):
            trc_sigma += np.sum(np.square(c[i, :]))
        return trc_sigma

    def _compute_trace_sqrt_mat(self, c_fake):
        mat_1 = np.matmul(np.transpose(c_fake), self.c_real)
        mat_2 = np.matmul(np.transpose(self.c_real), c_fake)
        mat_3 = np.matmul(mat_1, mat_2)
        eig, _ = np.linalg.eig(mat_3)
        trc = np.sum(np.sqrt(eig))
        return trc
