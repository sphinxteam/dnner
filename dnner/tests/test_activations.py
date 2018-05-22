import unittest

import numpy as np

import dnner
from dnner.activations import ReLU, LeakyReLU, Linear


class ActivationsTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ReLU(self):
        ac = ReLU(0.01).set_mode(is_output=False)

        i_exp, a_exp, v_exp = [0.973975, 1.408574, 0.038401]
        self.assertAlmostEqual(i_exp, ac.eval_i(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(a_exp, ac.iter_a(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(v_exp, ac.iter_v(10, 0.25, 0.5), places=6)

    def test_LeakyReLU_ReLU(self):
        ac = LeakyReLU(0.01, 0).set_mode(is_output=False)

        i_exp, a_exp, v_exp = [0.973975, 1.408574, 0.038401]
        self.assertAlmostEqual(i_exp, ac.eval_i(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(a_exp, ac.iter_a(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(v_exp, ac.iter_v(10, 0.25, 0.5), places=6)

    def test_LeakyReLU_Linear(self):
        ac = LeakyReLU(0.01, 1).set_mode(is_output=False)
        ac_test = Linear(0.01).set_mode(is_output=False)

        i_exp, a_exp, v_exp = [ac_test.eval_i(10, 0.25, 0.5),
                               ac_test.iter_a(10, 0.25, 0.5),
                               ac_test.iter_v(10, 0.25, 0.5)]

        self.assertAlmostEqual(i_exp, ac.eval_i(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(a_exp, ac.iter_a(10, 0.25, 0.5), places=6)
        self.assertAlmostEqual(v_exp, ac.iter_v(10, 0.25, 0.5), places=6)


if __name__ == "__main__":
        unittest.main()
