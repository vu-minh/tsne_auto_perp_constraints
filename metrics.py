# some metric measurement for DR methods

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
from sklearn.isotonic import IsotonicRegression

MACHINE_EPSILON = np.finfo(np.double).eps


class DRMetric(object):
    """ Metric measurements for DR methods
    """

    def __init__(self, X, Y):
        """ Create Metric object
        Args:
            X (ndarray): input data in high dimensional space
            Y (ndarray): embedded result in low dimensional space
        """
        super(DRMetric, self).__init__()

        # pre-calculate pairwise distance in high-dim and low-dim
        self.dX = pdist(X, "euclidean")
        self.dY = pdist(Y, "euclidean")

    def _qnx(self, a, b):
        """Vectorized version of `self._Qnx` for all values of `k`
        """
        n = len(a)
        common = set()
        res = []
        for k in range(1, n - 1):
            common |= {a[k], b[k]}
            q = (2 * k - len(common)) / (k * n)
            assert 0 <= 1 <= 1
            res.append(q)
        return res

    def auc_rnx(self):
        """Vectorized version of `self._auc_rnx`
        """
        idX = np.argsort(squareform(self.dX**2), axis=1)
        idY = np.argsort(squareform(self.dY**2), axis=1)
        n = len(idX)

        qnx = [self._qnx(a, b) for a, b in zip(idX, idY)]
        qnx = np.sum(qnx, axis=0)

        ks = np.arange(1, n - 1)
        rnx = ((n - 1) * qnx - ks) / (n - 1 - ks)
        return (rnx / ks).sum() / (1.0 / ks).sum()

    def pearsonr(self):
        """Calculate Pearson correlation coefficient b.w. two vectors
            $$ \textnormal{CC} =
                \textnormal{pearson\_correlation}(d^x, d^y) =
                \frac{\textnormal{Cov}(d^x, d^y)}{\sigma(d^x)\sigma(d^y)}
            $$
        """
        p, _ = pearsonr(self.dX, self.dY)
        return p

    def cca_stress(self):
        """Curvilinear Component Analysis Stress function
            $$ \textnormal{CCA} = \sum_{ij}^{N}
                (d^{x}_{ij} - d^{y}_{ij})^2 F_{\lambda}(d^{y}_{ij})
            $$
            where $d^{x}_{ij}$ is pairwise distance in high-dim,
            $d^{y}_{ij}$ is pairwise distance in low-dim,
            $F_{\lambda}(d^{*}_{ij}$ is decreasing weighting-function.
            For CCA, there are some choises for weighting-function:
            e.g. step function (depends $\lambda$), exponential func or
            $F(d^{y}_{ij}) = 1 - sigmoid(d^{y}_{ij}$.
        """
        dX = scale(self.dX)
        dY = scale(self.dY)
        diff = dX - dY
        weight = 1.0 - 1.0 / (1.0 + np.exp(-dY))
        stress = np.dot(diff**2, weight)
        return stress

    def mds_isotonic(self):
        """Stress function of MDS
            + Pairwise distances vector in high-dim is fitted into an
            Isotonic Regression model
            + The stressMDS function is then applied for the isotonic-fitted
            vector and the pairwise distance vector in low-dim
            $$ \textnormal{nMDS} = \sqrt{\frac
                { \sum_{ij} (d^{iso}_{ij} - d^{y}_{ij})^2 }
                { \sum_{ij} d^{y}_{ij} } }
            $$
            where $d^{y}_{ij}$ is pairwise distance in low-dim.
        """
        dX = scale(self.dX)
        dY = scale(self.dY)
        ir = IsotonicRegression()
        dYh = ir.fit_transform(X=dX, y=dY)
        return norm(dYh - dY) / norm(dY)

    def sammon_nlm(self):
        """Stree function for Sammon Nonlinear mapping
            $ \textnormal{NLM} = \frac{1}{\sum_{ij} d^{x}_{ij}}
                \sum_{ij} \frac{ (d^{x}_{ij} - d^{y}_{ij})^2 }{d^{x}_{ij}]}
            $
        """
        dX = self.dX / np.std(self.dX)
        dX_inv = np.divide(1.0, dX,
                           out=np.zeros_like(dX), where=(dX != 0))
        dY = self.dY / np.std(self.dY)
        diff = dX - dY
        stress = np.dot((diff ** 2), dX_inv)
        return stress / dX.sum()

    def _Qnx(self, k):
        """Calculate $Q_{NX}(k)= \\
          \frac{1}{Nk} \sum_{i=1}^{N} |v_{i}^{k} \cap n_{i}^{k}| $
        Args:
            k (int): number of neighbors
        Returns:
            float: value of Q
        """
        assert 1 <= k <= self.n_samples - 1

        Vk = self.idX[:, :k]
        Nk = self.idY[:, :k]
        q_nx = sum([np.intersect1d(a, b, assume_unique=True).size
                    for a, b in zip(Vk, Nk)])
        q_nx /= (k * self.n_samples)

        assert 0.0 <= q_nx <= 1.0
        return q_nx

    def _Rnx(self, k):
        """Calculate rescaled version of $Q_{NX}(k)$
          $R_{NX}(k) =  \frac{(N-1) Q_{NX}(k) - k}{N - 1 - k} $
        Args:
            k (int): number of neighbors
        Returns:
            float: value of R
        """
        assert 1 <= k <= self.n_samples - 2
        rnx = (self.n_samples - 1) * self._Qnx(k) - k
        rnx /= (self.n_samples - 1 - k)
        return rnx

    def _auc_rnx(self):
        """Calculate Area under the $R_{NX}(k)$ curve in the log-scale of $k$
            $$ \textnormal{AUC}_{log}\textnormal{RNX} =
                \frac {\left(\sum_{k=1}^{N-2} \frac{R_{NX}(k)}{k} \right)}
                      {\left(\sum_{k=1}^{N-2}\frac{1}{k}\right)}
            $$
        """
        auc = sum([self._Rnx(k) / k for k in range(1, self.n_samples - 1)])
        norm_const = sum([1 / k for k in range(1, self.n_samples - 1)])
        auc /= norm_const
        assert 0.0 <= auc <= 1.0
        return auc
