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

    def predict(self, input_vec):
        list1 = [x * w for x,w in zip(input_vec, self.weights)] 
        val = reduce(lambda x, y: x + y , list1, 0.0)
        output = self.activator(val + self.bias)
        return output

    def train(self, input_vecs, labels, iteration, rate):

        # 负责迭代，
        for i in range(iteration):
            # 一次迭代的过程
            samples = zip(input_vecs, labels)

            # 开始一次迭代
            for (input_vec, label) in samples:
                # 输出这一次计算出的权重weights
                output = self.predict(input_vec=input_vec)

                # 更新权重
                delta = label - output
                self.weights = [w + rate * delta * x for x,w in zip(input_vec, self.weights)]
                
                # 更新bias
                self.bias += rate * delta


if __name__ == "__main__":
    input_num = 2
    activator = lambda x : (1 if x > 0 else 0)

    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    iteration = 10
    rate = 0.1

    p = Perceptron(input_num=input_num, activator=activator)
    p.train(input_vecs=input_vecs, labels=labels, iteration=iteration, rate=rate)
    print(p)
    print ('1 and 1 = %d' % p.predict([1, 1]))
    print ('0 and 0 = %d' % p.predict([0, 0]))
    print ('1 and 0 = %d' % p.predict([1, 0]))
    print ('0 and 1 = %d' % p.predict([0, 1]))


    


    