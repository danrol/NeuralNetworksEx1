import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# method that builds xor neural network. Method returns truth table result of XOR and average square error
def build_xor_neural_network(matrix, excepted, k, predefined_w1, predefined_w2, hidden_layer_biases,
                             output_neuron_bias):
    amount_input_neurons = 2
    amount_output_neurons = 1
    bypass = amount_input_neurons + k #used in case hidden layer build only from one layer
    temperature = 0.001

    # define placeholder that will be used later after tensorflow session starts
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])

    w1 = tf.Variable(predefined_w1.reshape([amount_input_neurons, k]), dtype=tf.float32)

    if k == 1:
        w2 = tf.Variable(predefined_w2.reshape([bypass, amount_output_neurons]), dtype=tf.float32)

    else:
        w2 = tf.Variable(predefined_w2.reshape([k, amount_output_neurons]), dtype=tf.float32)

    b1 = tf.Variable(hidden_layer_biases, dtype=tf.float32)
    b2 = tf.Variable(output_neuron_bias, tf.float32)

    z1 = tf.matmul(x, w1) + b1
    hidden_layer_res = tf.sigmoid(z1 / temperature)

    if k == 1:
        conc_hidden_layer_res = tf.concat([hidden_layer_res, x], 1)
    else:
        conc_hidden_layer_res = hidden_layer_res

    z2 = tf.matmul(conc_hidden_layer_res, w2) + b2

    final_output = tf.sigmoid(z2 / temperature)

    squared = tf.square(final_output - y)
    loss = tf.reduce_sum(squared)
    var = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(var)
    result = sess.run([final_output, loss], {x: matrix, y: excepted})
    return result

def write_output_to_file(result_str, k, text_file):
    (truth_table_result, avg_square_error) = result
    result_str = f'k = {k} neural network truth table result  = \n{truth_table_result}, \n' \
             f'expected result = {expected_results}, loss = {avg_square_error}\n\n'
    print(result_str)
    text_file.write(result_str)

if __name__ == '__main__':
    input_train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_results = [[0], [1], [1],
                        [0]]  # expected results from activating XOR (corresponds to the lists in input_train list)

    ##### Start of block Of code dealing with number of hidden neurons equal to 4 ####
    k = 4
    predefined_w1 = np.array([-1., -1., 1., 1., -1., 1., -1., 1.])
    predefined_w2 = np.array([-1, 1, 1, -1])
    biases = [-0.5, -0.5, -0.5, -2.5]
    output_neuron_bias = [-0.5]
    print(f'k = {k} \nweights between inputs to hidden layer = {predefined_w1}'
          f'\nweights between hidden layer to output layer = {predefined_w2}\nhidden layer biases = {biases}'
          f'\noutput layer bias = {output_neuron_bias}')
    result = build_xor_neural_network(input_train_x, expected_results, k, predefined_w1,
                                                                      predefined_w2, biases, output_neuron_bias)
    text_file = open("output.txt", "w")
    write_output_to_file(result, k, text_file)
    ##### End of block Of code dealing with number of hidden neurons equal to 4 ####

    ##### Start of block Of code dealing with number of hidden neurons equal to 2 ####
    k = 2
    predefined_w1 = np.array([1, -1, 1, -1])
    predefined_w2 = np.array([1, 1])
    biases = [-0.5, 1.5]
    output_neuron_bias = [-1.5]
    print(f'k = {k} \nweights between inputs to hidden layer = {predefined_w1}'
          f'\nweights between hidden layer to output layer = {predefined_w2}\nhidden layer biases = {biases}'
          f'\noutput layer bias = {output_neuron_bias}')
    result = build_xor_neural_network(input_train_x, expected_results, k, predefined_w1,
                                      predefined_w2, biases, output_neuron_bias)
    write_output_to_file(result, k, text_file)
    ##### End of block Of code dealing with number of hidden neurons equal to 2 ####

    ##### Start of block Of code dealing with number of hidden neurons equal to 1 ####
    k = 1
    predefined_w1 = np.array([1, 1])
    predefined_w2 = np.array([-2, 1, 1])
    biases = [-1.5]
    output_neuron_bias = [-0.5]
    print(f'k = {k} \nweights between inputs to hidden layer = {predefined_w1}'
          f'\nweights between hidden layer to output layer = {predefined_w2}\nhidden layer biases = {biases}'
          f'\noutput layer bias = {output_neuron_bias}')
    result = build_xor_neural_network(input_train_x, expected_results, k, predefined_w1,
                                      predefined_w2, biases, output_neuron_bias)
    write_output_to_file(result, k, text_file)
    ##### End of block Of code dealing with number of hidden neurons equal to 1 ####
    text_file.close()