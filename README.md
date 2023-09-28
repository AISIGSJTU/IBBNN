<div align="center">
  
# Information Bound and its Applications in Bayesian Neural Networks
[![Conference](http://img.shields.io/badge/ECAI-2023-4b44ce.svg)](https://ecai2023.eu/) 

</div>

## Use 
Run `main.py` to train models with IB regularization and IB variance regularization. 
The parameter `ratio` and `lambda2` corresponding to $\lambda_1$ and $\lambda_2$ in Eq. (16) and Eq. (17).
Note that please modify the file path to your own path accordingly.

## Core Codes

The core codes to estimate Information Bound are implemented in each layer, e.g., `models/layers/conv2d.py`, line 100-124:
```python
def kl_output(self, mean_batch=True):
    batch_size = self.input.shape[0]
    input = self.input
    sig_weight = torch.exp(self.sigma_weight)
    if self.mu_bias is not None:
        sig_bias = torch.exp(self.sigma_bias)
    else:
        sig_bias = None

    mu_out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride, self.padding, self.dilation,
                      self.groups)
    sig_out = F.conv2d(input.pow(2), sig_weight.pow(2), sig_bias.pow(2), self.stride, self.padding, self.dilation,
                       self.groups).pow(0.5)
    if mean_batch:
        kl_out = (- torch.log(sig_out) + 0.5 * (sig_out.pow(2) + mu_out.pow(2)) - 0.5).mean()
    else:
        kl_out = (- torch.log(sig_out) + 0.5 * (sig_out.pow(2) + mu_out.pow(2)) - 0.5).reshape(batch_size, -1)
        kl_out = kl_out.mean(dim=1)
        return kl_out
    if torch.isinf(kl_out):
        logging.error("INF!")
        logging.debug(kl_out)
        logging.debug(sig_out)
        raise RuntimeError(kl_out, sig_out, mu_out)
    return kl_out
```

## Cite
TBD
