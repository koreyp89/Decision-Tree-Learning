import collections
import math
import sys


def calc_entropy(prob):
    return 0 if prob == 0 else -1 * prob * math.log(prob, 2)


def get_plurality_winners(values, examples):
    counter = {i: 0 for i in values}
    for example in examples:
        if example["class"] in counter.keys():
            counter[example["class"]] += 1
    x = max(counter, key=counter.get)
    return [i for i in counter.keys() if counter[i] == counter[x]]


def all_same(examples):
    value = examples[0]["class"]
    for example in examples:
        if example["class"] != value:
            return False
    return True


def get_all_classes(examples):
    classes = {}
    for example in examples:
        if example["class"] not in classes.keys():
            classes[example["class"]] = 1
        else:
            classes[example["class"]] += 1
    return classes


def calc_total_entropy(examples):
    total_entropy = 0
    classes = get_all_classes(examples)
    for c in classes.keys():
        prob = classes[c] / len(examples)
        total_entropy += calc_entropy(prob)
    return total_entropy


def get_examps(examples, attribute, value):
    return [i for i in examples if i[attribute] == value]


def get_all_values(attribute, examples):
    values = list({i[attribute] for i in examples})
    return values


def calculate_attribute_info_gain(attributes, examples):
    attribute_info_gain = {}
    examples_entropy = calc_total_entropy(examples)
    for attribute in attributes:
        summation = 0
        values = get_all_values(attribute, examples)
        for value in values:
            examps = [i for i in examples if i[attribute] == value]
            summation += (len(examps) / len(examples)) * calc_total_entropy(
                examps)
        attribute_info_gain[attribute] = examples_entropy - summation
    return attribute_info_gain


def most_important(attributes, info_gain):
    most_important_attribute = attributes[0]
    for attribute in attributes:
        if info_gain[attribute] > info_gain[most_important_attribute]:
            most_important_attribute = attribute
    return most_important_attribute


def out_of_attributes(values, examples, parents):
    winners = get_plurality_winners(values, examples)
    return min(get_plurality_winners(values, parents)) if len(winners) > 1 else min(winners)


class decision_tree:

    def __init__(self, training_file, test_file):
        self.attributes = []
        self.classes = []
        self.training_set = []
        self.test_set = []
        self.create_data_set(training_file)
        self.create_test_set(test_file)
        self.classes = get_all_classes(self.training_set)
        self.tree = self.learn_decision_tree(self.training_set,
                                             [i for i in self.attributes if i != "class"])
        self.print_tree(self.tree, 0)
        print()
        self.test_train_set()
        print()
        self.test_test_set()

    def create_data_set(self, t_file):
        training_file = open(t_file)
        data = training_file.read()
        lines = data.splitlines()
        attribute_line = lines.pop(0)
        self.attributes = attribute_line.split()
        for line in lines:
            values = line.split()
            data_example = {}
            for i in range(len(self.attributes)):
                data_example[self.attributes[i]] = values[i]
            self.training_set.append(data_example)
        training_file.close()

    def create_test_set(self, t_file):
        test_file = open(t_file)
        data = test_file.read()
        lines = data.splitlines()
        lines = lines[1:]
        for line in lines:
            values = line.split()
            data_example = {}
            for i in range(len(self.attributes)):
                data_example[self.attributes[i]] = values[i]
            self.test_set.append(data_example)

    def print_tree(self, tree, depth):
        od = collections.OrderedDict(sorted(tree[1].items()))
        for value in od.keys():
            for i in range(depth):
                print("| ", end='')
            if isinstance(od[value], str):
                print(tree[0], "=", value, ":", od[value])
            else:
                print(tree[0], "=", value, ":")
                self.print_tree(od[value], depth + 1)

    def learn_decision_tree(self, examples, attributes):
        if all_same(examples):
            return examples[0]["class"]
        if not attributes:
            return out_of_attributes(get_all_values("class", examples), examples, self.training_set)
        most_important_attribute = most_important(attributes,
                                                  info_gain=calculate_attribute_info_gain(attributes, examples))
        tree = [most_important_attribute, {}]
        for value in ["0", "1", "2"]:
            exs = [i for i in examples if i[most_important_attribute] == value]
            tree[1][value] = \
                self.learn_decision_tree(exs,
                                         [i for i in attributes if i != most_important_attribute]) if exs \
                else min(get_plurality_winners(get_all_values("class", self.training_set), self.training_set))
        return tree

    def test_test_set(self):
        num_correct = 0
        for example in self.test_set:
            prediction = self.predict(self.tree, example)
            num_correct += 1 if prediction == example["class"] else 0
        print("Accuracy on test set (", len(self.test_set), " instances): ",
              round(100 * num_correct / len(self.test_set), 1), "%", sep='')

    def test_train_set(self):
        num_correct = 0
        for example in self.training_set:
            prediction = self.predict(self.tree, example)
            num_correct += 1 if prediction == example["class"] else 0
        print("Accuracy on training set (", len(self.training_set), " instances): ",
              round(100 * num_correct / len(self.training_set), 1), "%", sep='')

    def predict(self, tree, example):
        return tree[1][example[tree[0]]] if isinstance(tree[1][example[tree[0]]], str) else \
            self.predict(tree[1][example[tree[0]]], example)


decision_tree(sys.argv[1], sys.argv[2])
