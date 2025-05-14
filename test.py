import torch
import pointwise_compiler

A = torch.randn(1024)
B = torch.randn(1024)

@torch.jit.script
def foo(a, b):
  c = a.mul(b)
  a = c.mul(c)
  a = c.mul(a)
  return a

print(foo.graph_for(A,B))

