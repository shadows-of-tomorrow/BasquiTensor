import numpy as np
import scipy as sp
from gans.utils import generate_real_samples
from gans.utils import generate_fake_samples
from tensorflow.keras.applications.inception_v3 import InceptionV3


class FIDCalculator:

    def __init__(self, image_processor, fid_res=299, m=128, n=10000, d=2048):
        print(f"Constructing Frechet Inception Distance calculator...")
        self.image_processor = image_processor
        self.m, self.n, self.d = m, n, d
        self.img_shape = (fid_res, fid_res)
        self.fid_shape = (fid_res, fid_res, 3)
        self.inception_network = InceptionV3(include_top=False, pooling='avg', input_shape=self.fid_shape)
        self.mu_real, self.cov_real, self.trc_cov_real = self._precompute_values()

    def compute_fast_fid(self, generator):
        # 1. Generate fake images.
        x_fake, _ = generate_fake_samples(generator, generator.input_shape[1], self.m)
        x_fake = self.image_processor.resize_np_array(x_fake, self.img_shape)
        # 2. Compute activations on fake images.
        a_fake = np.transpose(self.inception_network.predict(x_fake))
        mu_fake = a_fake.mean(axis=1).reshape(-1, 1)
        # 3. Compute the trace of the fake covariance matrix.
        c_fake = self._compute_c_matrix(a_fake, mu_fake, self.m)
        cov_fake = np.matmul(c_fake, np.transpose(c_fake))
        trc_cov_fake = np.trace(cov_fake)
        # 4. Compute the trace of the "square root" matrix.
        trc_sqrt_mat = self._compute_trace_sqrt_mat(c_fake)
        # 6. Compute difference between means.
        mu_diff = np.sum(np.square(self.mu_real-mu_fake))
        # 5. Compute FID by adding components.
        fid = mu_diff + self.trc_cov_real + trc_cov_fake - 2.0 * trc_sqrt_mat
        return fid

    def compute_fid(self, generator):
        # 1. Generate real and fake images.
        x_real, _ = generate_real_samples(self.image_processor, self.n, self.img_shape)
        x_fake, _ = generate_fake_samples(generator, generator.input_shape[1], self.m)
        # 2. Calculate activations.
        a_real = self.inception_network.predict(x_real)
        a_fake = self.inception_network.predict(x_fake)
        # 3. Calculate mean and covariance matrix.
        mu_real, sigma_real = a_real.mean(axis=0), np.cov(a_real, rowvar=False)
        mu_fake, sigma_fake = a_fake.mean(axis=0), np.cov(a_fake, rowvar=False)
        # 4. Compute mean difference and square-root of covariance product.
        mu_diff = np.sum(np.square(mu_real-mu_fake))
        cov_diff = sp.linalg.sqrtm(sigma_real.dot(sigma_fake))
        # 5. Compute FID.
        if np.iscomplexobj(cov_diff):
            cov_diff = cov_diff.real
        return mu_diff + np.trace(sigma_real + sigma_fake - 2.0 * cov_diff)

    def _precompute_values(self):
        x_real, _ = generate_real_samples(self.image_processor, self.n, self.img_shape)
        a_real = self.inception_network.predict(x_real)
        mu_real, cov_real = a_real.mean(axis=0).reshape(-1, 1), np.cov(a_real, rowvar=False)
        trc_cov_real = np.trace(cov_real)
        return mu_real, cov_real, trc_cov_real

    def _compute_trace_sqrt_mat(self, c_fake):
        mat_m = np.matmul(np.transpose(c_fake), np.matmul(self.cov_real, c_fake))
        eig_values, _ = np.linalg.eig(mat_m)
        trc = np.sum(np.abs(np.sqrt(eig_values+10e-8)))
        return trc

    @staticmethod
    def _compute_c_matrix(a, mu, n_examples):
        return (1.0 / np.sqrt(n_examples-1)) * (a - np.matmul(mu, np.ones((1, n_examples))))
