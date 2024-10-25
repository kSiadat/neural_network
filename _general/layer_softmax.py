from numpy import empty, exp, zeros


class Layer_softmax:
    def __init__(self, ignore_1=None, ignore_2=None, ignore_3=None, ignore_4=None, load_text=None):
        pass

    def load(self):
        pass

    def save(self):
        return ""

    def evaluate(self, inp):
        exp_arr = exp(inp)
        exp_sum = exp_arr.sum()
        self.output = empty(len(inp))
        for x in range(len(inp)):
            self.output[x] = exp_arr[x] / exp_sum
        return self.output

    def backpropagate(self, inp, gradient):
        d = zeros(len(inp))
        for x in range(len(inp)):
            for y in range(len(inp)):
                if x == y:
                    print(x, y)
                    print(self.output[x] * (1 - self.output[x]))
                    d[x] += self.output[x] * (1 - self.output[x])
                    print(d[x])
                else:
                    print(x, y)
                    print(-self.output[x] * self.output[y])
                    d[x] += -self.output[x] * self.output[y]
                    print(d[x])
        print(f"d:\n{d}")
        print(f"gradient 1:\n{gradient}")
        d = d * gradient
        return d
