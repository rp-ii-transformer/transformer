import numpy as np

class Adam:
    def __init__(self, params, lr=0, betas=(0.9,0.98), eps=1e-9):
        self.params = params  # lista de arrays NumPy
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_num = 0
        self.lr = lr

    def step(self, grads):
        self.step_num += 1
        lr_t = self.lr * min(self.step_num**-0.5,
                             self.step_num * self.lr**-1.5)
        updated = []
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*g
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(g**2)
            m_hat = self.m[i] / (1-self.beta1**self.step_num)
            v_hat = self.v[i] / (1-self.beta2**self.step_num)
            p -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
            updated.append(p)
        return updated


def noam_schedule(d_model, warmup_steps, step):
    return (d_model**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)
