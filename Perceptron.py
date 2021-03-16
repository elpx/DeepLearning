from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator) -> None:
        """
        activator -> f
        weights   -> w
        bias      -> b
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self) -> str:
        return "weights: %s, bias: %s" % (self.weights, self.bias)

    def train(self, input_vecs, labels, iteration, rate):

        # 负责迭代，
        for i in range(iteration):
            # 一次迭代的过程
            samples = zip(input_vecs, labels)

            # 开始一次迭代
            for (input_vec, label) in samples:
                print("input_vec: ", input_vec, "label: ", label, "weights: ", self.weights)

                # 输出这一次计算出的权重weights
                zip1 = zip(input_vec, self.weights)
                print("zip1: ", list(zip1))

                # map1 = map (lambda (x, w): x * w, zip1)
                # map1 = map(lambda x: x[0] * x[1], zip1)
                list1 = [x * w for x,w in zip1] #这里用zip1就会出问题
                print("list1: ", list1)

                val = reduce(lambda x, y: x + y , list1, 0.0)
                print("val: ", val)

                output = self.activator(val + self.bias)
                print("bias: ", self.bias)
                print("output: ", output)

                # 更新权重
                delta = label - output
                zip2 = zip(input_vec, self.weights)
                self.weights = list(map(lambda x: x[1] + rate * delta * x[0], zip2))
                
                # 更新bias
                self.bias += rate * delta

                print("\n")


if __name__ == "__main__":
    input_num = 2
    activator = lambda x : (1 if x > 0 else 0)

    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    iteration = 10
    rate = 0.1

    p = Perceptron(input_num=input_num, activator=activator)
    p.train(input_vecs=input_vecs, labels=labels, iteration=iteration, rate=rate)

    


    