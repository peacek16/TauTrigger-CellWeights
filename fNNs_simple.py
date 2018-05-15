import tensorflow as tf
import random
import numpy as np
import math



"""
     File ouputs: a flat file with weights of each tensor cells and acc = net-true/true  lines by lines 
     Changed variables: INPUT_DIMENSION, string names of dataset and fileout which could be from a single layer(L2) or multiple layers 
     Environment settings: run on environment with Tensorflow such as Tier3. 
                           need a flat file containning Et cells to train the simple NN 
     Credit: https://stackoverflow.com/questions/38319898/tensorflow-neural-net-with-continuous-floating-point-output?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa 

"""
INPUT_DIMENSION = 85*5
OUTPUT_DIMENSION = 1
TRAINING_RUNS = 300
BATCH_SIZE = 500 
VERF_SIZE =int(0.25*500)

dataset = np.loadtxt("LEtTau_ViEt15Et5Eta1.4_Train.data", delimiter=",")
np.random.shuffle(dataset)
fileout = open("LEtTau_ViEt15Et5Eta1.4_W.data","write")
# Generate two arrays, the first array being the inputs that need trained on, and the second array containing outputs.


# Generate a bunch of data points and then package them up in the array format needed by
# tensorflow
def generate_batch_data(num):
    xs = []
    ys = []
    np.random.shuffle(dataset)
    for i in range(num):
         xs.append(np.array(dataset[i][0:INPUT_DIMENSION-85])/1000)
         ys.append(np.array([dataset[i][INPUT_DIMENSION+1]])/1000)
         
    return (np.array(xs),np.array(ys))

# Define a single-layer neural net.  Originally based off the tensorflow mnist for beginners tutorial

# Create a placeholder for our input variable
x = tf.placeholder(tf.float32, [None,INPUT_DIMENSION-85])

# Create variables for our neural net weights and bias
W = tf.Variable(tf.ones([INPUT_DIMENSION-85, OUTPUT_DIMENSION])) 

# Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
# tutorial, I have removed the softmax op.  My expectation is that net' will return a floating point
# value.
net = tf.matmul(x,W) 
#net = tf.reduce_sum(x*W+b, axis=1, name='out', keep_dims=True)

# Create a placeholder for the expected #result during training
expected = tf.placeholder(tf.float32, [None,OUTPUT_DIMENSION])

# Same training as used in mnist example
loss = tf.reduce_mean(tf.square(expected - net))
# cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1).minimize(loss)

sess = tf.InteractiveSession()

init = tf.initialize_all_variables()
sess.run(init)

# Perform our training runs

for i in range(TRAINING_RUNS):
    print("trainin run: ", i, )

    batch_inputs, batch_outputs = generate_batch_data(BATCH_SIZE)

    # I've found that my weights and bias values are always zero after training, and I'm not sure why.
    sess.run(train_step, feed_dict={x: batch_inputs, expected: batch_outputs})

    # Test our accuracy as we train...  I am defining my accuracy as the error between what I
    # expected and the actual output of the neural net.\
    #I = tf.ones(tf.shape(expected))
    #correct_prediction = tf.subtract(I,abs(tf.divide(tf.subtract(net,expected),expected)))
    correct_prediction = tf.divide(tf.subtract(expected,net),expected)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #accuracy = tf.subtract(expected, net)  # using just subtract since I made my verification size 1 for debug
    # tf.subtract()
    # Uncomment this to debug
    # import pdb; pdb.set_trace()

    #fileout.write(print(sess.run(W)))


    batch_inputs, batch_outputs = generate_batch_data(VERF_SIZE)
    acc = sess.run(accuracy, feed_dict={x: batch_inputs, expected: batch_outputs})
    weight = sess.run(W, feed_dict={x: batch_inputs, expected: batch_outputs})
    result = sess.run(net, feed_dict={x: batch_inputs, expected: batch_outputs})
    #result2 = accuracy.eval(feed_dict={x:batch_inputs, expected:batch_outputs})
    #print("      loss: ", loss.eval({x: batch_inputs, expected: batch_outputs}))
    #print("      expected:", batch_outputs)
    #print("       result:",result)
    print("      accuracy: ", acc)

    
    ls_weight = list(weight)
    fileout.write(','.join(str(ls_weight[i][0]) for i in range(len(ls_weight))))
    fileout.write(','+str(acc))
    fileout.write("\n")

fileout.close()
