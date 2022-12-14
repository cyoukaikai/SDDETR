import torch

# x1, y1, w, h
a = torch.tensor([[1.6900e+00, 1.6900e+00, 5.0000e+02, 3.6826e+02],
        [1.0592e+02, 4.0000e-02, 1.8303e+02, 5.5120e+01],
        [1.3745e+02, 4.4250e+01, 4.2438e+02, 9.3160e+01],
        [1.2115e+02, 2.0642e+02, 1.7820e+02, 2.7045e+02],
        [1.7125e+02, 1.8627e+02, 2.6698e+02, 2.5740e+02],

        [4.9410e+01, 1.6208e+02, 1.0324e+02, 2.0708e+02],
        [1.9105e+02, 2.5174e+02, 2.5937e+02, 3.1880e+02],
        [2.5853e+02, 1.8971e+02, 2.9647e+02, 2.6206e+02],
        [1.7070e+02, 8.3140e+01, 2.0519e+02, 1.3620e+02],
        [1.0853e+02, 8.5590e+01, 1.6208e+02, 1.2290e+02],
        [1.4466e+02, 4.4720e+01, 4.2256e+02, 8.8880e+01],
        [2.1199e+02, 1.3621e+02, 2.7881e+02, 1.9771e+02]])

a_area = a[:, 2] * a[:, 3]
order = torch.sort(a_area)

b = torch.tensor([[3.6031e+00, 3.6053e+00, 1.0660e+03, 7.8562e+02],
        [2.2582e+02, 8.5333e-02, 3.9022e+02, 1.1759e+02],
        [2.9304e+02, 9.4400e+01, 9.0478e+02, 1.9874e+02],
        [2.5829e+02, 4.4036e+02, 3.7992e+02, 5.7696e+02],
        [3.6510e+02, 3.9738e+02, 5.6920e+02, 5.4912e+02],
        [1.0534e+02, 3.4577e+02, 2.2011e+02, 4.4177e+02],
        [4.0732e+02, 5.3705e+02, 5.5298e+02, 6.8011e+02],
        [5.5119e+02, 4.0471e+02, 6.3207e+02, 5.5906e+02],
        [3.6393e+02, 1.7737e+02, 4.3747e+02, 2.9056e+02],
        [2.3139e+02, 1.8259e+02, 3.4555e+02, 2.6219e+02],
        [3.0842e+02, 9.5403e+01, 9.0090e+02, 1.8961e+02],
        [4.5196e+02, 2.9058e+02, 5.9442e+02, 4.2178e+02]])


b_area = (b[:,2] - b[:, 0] ) * (b[:, 3] - b[:,1])
b_order = torch.sort(b_area)

print(f'a_order = {order}, b_order = {b_order}')