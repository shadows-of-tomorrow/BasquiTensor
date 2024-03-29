import numpy as np
import scipy as sp
import tensorflow as tf
from networks.utils.sampling import generate_real_images, generate_fake_images
from tensorflow.keras.applications.inception_v3 import InceptionV3


class FIDCalculator:

    def __init__(self, image_processor, fid_res=128, n_fake_samples=128, n_real_samples=5000, n_activations=2048):
        print(f"Constructing FID calculator...")
        self.image_processor = image_processor
        self.n_fake_samples = n_fake_samples
        self.n_real_samples = n_real_samples
        self.n_activations = n_activations
        self.chunk_size = 100
        self.img_shape = (fid_res, fid_res)
        self.fid_shape = (fid_res, fid_res, 3)
        self.inception_network = InceptionV3(include_top=False, pooling='avg', input_shape=self.fid_shape)
        self.mu_real, self.cov_real, self.trc_cov_real = self._precompute_reals()

    def compute_fast_fid(self, generator):
        # 1. Generate fake images.
        x_fake = generate_fake_images(
            self.image_processor,
            generator,
            self.n_fake_samples,
            self.img_shape,
            transform_type="min_max_to_zero_eager"
        )
        # 2. Compute activations on fake images.
        a_fake = np.transpose(self.inception_network.predict(x_fake))
        mu_fake = a_fake.mean(axis=1).reshape(-1, 1)
        # 3. Compute the trace of the fake covariance matrix.
        c_fake = self._compute_c_matrix(a_fake, mu_fake, self.n_fake_samples)
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
        x_real = generate_real_images(self.image_processor, self.n_real_samples, self.img_shape)
        x_fake = generate_fake_images(self.image_processor, generator, self.n_fake_samples, self.img_shape)
        # 2. Calculate activations.
        a_real = self.inception_network.predict(x_real)
        a_fake = self.inception_network.predict(x_fake)
        # 3. Calculate mean and covariance matrix.
        mu_real, cov_real = a_real.mean(axis=0), np.cov(a_real, rowvar=False)
        mu_fake, cov_fake = a_fake.mean(axis=0), np.cov(a_fake, rowvar=False)
        # 4. Compute mean difference and square-root of covariance product.
        mu_diff = np.sum(np.square(mu_real-mu_fake))
        cov_diff = sp.linalg.sqrtm(cov_real.dot(cov_fake))
        # 5. Compute FID.
        if np.iscomplexobj(cov_diff):
            cov_diff = cov_diff.real
        return mu_diff + np.trace(cov_real + cov_fake - 2.0 * cov_diff)

    def _precompute_reals(self):
        mu_real, cov_real = self._cov_mean_low_memory()
        trc_cov_real = np.trace(cov_real)
        return mu_real, cov_real, trc_cov_real

    def _cov_mean_low_memory(self):
        n_steps = self.n_real_samples // self.chunk_size
        mu_loop = np.zeros((self.n_activations, 1))
        cov_loop = np.zeros((self.n_activations, self.n_activations))
        for _ in range(n_steps):
            x_real = generate_real_images(
                self.image_processor,
                self.chunk_size,
                self.img_shape,
                transform_type="old_to_zero_one"
            )
            a_real = self.inception_network.predict(x_real)
            for k in range(self.chunk_size):
                a_real_k = a_real[[k], :].reshape(-1, 1)
                mu_loop += a_real_k
                cov_loop += np.outer(a_real_k, a_real_k)
        mu_loop /= (self.n_real_samples-1)
        cov_loop /= (self.n_real_samples-1)
        cov_loop -= np.outer(mu_loop, mu_loop)
        return mu_loop, cov_loop

    def _compute_trace_sqrt_mat(self, c_fake):
        mat_m = np.matmul(np.transpose(c_fake), np.matmul(self.cov_real, c_fake))
        eig_values, _ = np.linalg.eig(mat_m)
        trc = np.sum(np.abs(np.sqrt(np.abs(eig_values))))
        return trc

    @staticmethod
    def _compute_c_matrix(a, mu, n_examples):
        return (1.0 / np.sqrt(n_examples-1)) * (a - np.matmul(mu, np.ones((1, n_examples))))

