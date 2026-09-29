import os
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
brain_data = os.path.join(path, 'brain_data.npz')

data =np.load(brain_data, allow_pickle=True)
W = data['W']
sides = data['sides']
cols = data['cols']


def build_vector(left, right, center):
    left_in = left + center
    right_in = right + center

    x = np.where(sides == 'L', left_in, right_in)
    out = np.dot(W, x)
    return out[0], out[1]

def main():
    print(build_vector(0.5, 0, 0).shape)
    

if __name__ == "__main__":
    main()