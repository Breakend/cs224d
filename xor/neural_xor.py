import argparse
from nn.math import sigmoid
from numpy import *
from matplotlib.pyplot import *

# The goal of this is to demonstrate how to set weights of a two layer neural net to
# simulate an XOR function

# matplotlib settings
matplotlib.rcParams['savefig.dpi'] = 100

# arg parser whether to show the graphs or just save them
parser = argparse.ArgumentParser(description='Run through and manually set weights for a network.')
parser.add_argument('--data_graphs', help='Show initial demonstrative data graphs.')
args = parser.parse_args()

# data display functionality
colors = 'rbcm'
markers = 'soos'
def show_pts(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[0,idx], data[1,idx],
            marker=markers[i], linestyle='.',
            color=colors[i], alpha=0.5)
    gca().set_aspect('equal')

def show_pts_1d(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[idx], marker=markers[i], linestyle='.',
            color=colors[i], alpha=0.5)
    gca().set_aspect(npts/4.0)

# Sample data to demonstrate what kind of data we're talking about
npts = 4 * 40; random.seed(10)
x = random.randn(npts)*0.1 + array([i & 1 for i in range(npts)])
y = random.randn(npts)*0.1 + array([(i & 2) >> 1 for i in range(npts)])
data = vstack([x,y])

# if we want to show the initial data graphs then show them
if args.data_graphs:
    fig = figure(figsize=(4,4)); show_pts(data); ylim(-0.5, 1.5); xlim(-0.5, 1.5)
    xlabel("x"); ylabel("y"); title("Input Data")

    # How does weight affect sigmoid function? Here's a plot
    x = linspace(-1, 1, 100); figure(figsize=(4,3))
    plot(x, sigmoid(x), 'k', label="$\sigma(x)$");
    plot(x, sigmoid(5*x), 'b', label="$\sigma(5x)$");
    plot(x, sigmoid(15*x), 'g', label="$\sigma(15x)$");
    legend(loc='upper left'); xlabel('x');

    show()

# Here are the weights/biases
W = zeros((2,2))
b1 = zeros((2,1))
U = zeros(2)
b2 = 0

# Your task is to add weight functions to manually separate the data
#### REPLACE THESE IF YOU WANT TO MESS AROUND WITH WEIGHTS ####
W = array([[-1.0,-1.0],[1.0,1.0]])
b1 = array([[.5],[-1.5]])
U = array([1.0, 1.0])
b2 = -.5
z = 5
#### END REPLACING STUFF ####

# Feed-forward computation
h = sigmoid(z*(W.dot(data) + b1))
p = sigmoid(z*(U.dot(h) + b2))

# Plot hidden layer
subplot(1,2,1); show_pts(h)
title("Hidden Layer"); xlabel("$h_1$"); ylabel("$h_2$")
ylim(-0.1, 1.1); xlim(-0.1, 1.1)

# Save to temp file
tmpfile = '/tmp/hidden_layer.png'
savefig(tmpfile)
print "Hidden layer image saved to {}".format(tmpfile)

# Plot predictions
subplot(1,2,2); show_pts_1d(p)
title("Output"); ylabel("Prediction"); xticks([])
axhline(0.5, linestyle='--', color='k')

# Save to temp file
tmpfile = '/tmp/output_layer.png'
savefig(tmpfile)
print "Output layer image saved to {}".format(tmpfile)
