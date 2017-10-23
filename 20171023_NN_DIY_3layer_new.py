# Make your own neural network
# 2017/10 Seong-Hun Choe
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
#描画はNotebook内、外部のWindowではない
#matplotlib inline

# Designing the neural network with class

class neuralNetwork:
    # 初期化
    def __init__(self, inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodes2 = hiddennodes2 # 2nd hidden layer
        self.onodes = outputnodes

        # 重み行列　WihとWho
        # np.random.normal(分布の中心値, 分布の標準偏差),(返すサイズ))
        # pow(x,y) : x^y at matlab (べき乗）
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.whh2 = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes)) # 2nd hidden layer
        self.wh2o = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 学習率
        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 学習
    def train(self, inputs_list, targets_list):
        # 入力リストを行列に変換
        # np.array ndmin ->　アレイの最小限の次元を設定
        # np.array .T -> 行列の前置(Transpose）
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 隠れ層1の入出力信号の計算
        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        # 隠れ層2の入出力信号の計算
        hidden_inputs2 = np.dot(self.whh2,hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        # 出力層の入出力信号の計算
        final_inputs = np.dot(self.wh2o, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        # 誤差の計算
        output_errors = targets - final_outputs

        # 隠れ層の誤差＝出力層の誤差＊Who
        hidden_errors2 = np.dot(self.wh2o.T, output_errors)
        hidden_errors1 = np.dot(self.whh2.T, hidden_errors2)

        # Wh2oの重みの更新
        self.wh2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs2))
        # Whh2の重みの更新
        self.whh2 += self.lr * np.dot((hidden_errors2 * hidden_outputs2 * (1.0 - hidden_outputs2)),np.transpose(hidden_outputs1))
        # Wihの重みの更新
        self.wih += self.lr * np.dot((hidden_errors1 * hidden_outputs1 * (1.0 - hidden_outputs1)),np.transpose(inputs))
        pass

    # 推論
    def query(self, inputs_list):
        # 入力リストを行列に変換
        # np.array ndmin ->　アレイの最小限の次元を設定
        # np.array .T -> 行列の前置(Transpose）
        inputs = np.array(inputs_list, ndmin=2).T

        # 隠れ層1の入出力信号の計算
        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        # 隠れ層2の入出力信号の計算
        hidden_inputs2 = np.dot(self.whh2, hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        # 出力層の入出力信号の計算
        final_inputs = np.dot(self.wh2o, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        # 活性化関数はシグモイド関数
        # def sigmoid(self):
        # return 1.0 / (1.0+ np.exp(-self))


# Program


input_nodes = 784
hidden_nodes = 300
hidden_nodes2 = 300
output_nodes = 10

# 学習率
learning_rate = 0.2

# neural networkのinstance生成

n = neuralNetwork(input_nodes, hidden_nodes,hidden_nodes2, output_nodes, learning_rate)

# MNIST Data
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# neural networkの学習

epochs = 1

for e in range(epochs):

    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # targets
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)
    pass
pass

# test

test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#Neural Networkのテスト

#scorecardは判定のリスト、最初は空
scorecard = []

#テストデータのすべてのデータに対して実行
for record in test_data_list:

    all_values = record.split(',')
    correct_label = int(all_values[0])
    #print(correct_label, "correct_label")

    inputs = (np.asfarray(all_values[1:])/255 * 0.99)+0.01

    #NNへの照会
    outputs = n.query(inputs)

    #最大値のラベルに対応
    label = np.argmax(outputs)
    #print(label, "network's answer")

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

    pass

#print(scorecard)
scorecard_array= np.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)


