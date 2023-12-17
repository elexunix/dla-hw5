def init_weights(module, mu=0, sigma=0.01):
  if 'Conv' in module.__class__.__name__:
    module.weight.data.normal_(mu, sigma)


def get_padding(kernel_size, dilation):
  return (kernel_size - 1) * dilation // 2
