import numpy as np

# Data structure
class Filter:
    '''
    The filter of a conv layer has type 'Filter'.
    Both the filter and its bias are stored in it.
    The gradients of the filter and the bias are stored as an attribute, too.
    We require the filt to be a tensor and bias a number
    (eg. filt = np.array([[[1,2],[2,3]],[[2,3],[3,4]]])), bias = 0)
    '''
    def __init__(self, filt, bias):
        self.kernel = filt
        self.bias = bias
        self.kernelgrad = np.zeros(filt.shape)
        self.biasgrad = 0
        if filt.shape != ():
            self.size = filt.shape[1]

class Layer:
    '''
    Layer is a top class with some subclasses (ConvLayer, PoolLayer etc.)
    The attribute 'layer' is the tensor.
    'lastlayer' and 'nextlayer' should also be Layer type, which point to the adjoint layer of this layer.
    'grad' is the gradient of this layer
    'ltype' should be str type, the value of which consists of 'conv', 'pool', 'relu', 'softmax'
    '''
    def __init__(self, layer=np.array([[[0]]]), ltype='layer', lastlayer=0, nextlayer=0):
        self.layer = layer
        self.lastlayer = lastlayer
        self.nextlayer = nextlayer
        self.grad = np.zeros(layer.shape)
        self.layertype = ltype
        self.depth = layer.shape[0]
        self.width = layer.shape[1]

def scan(inp, filt, bias, padding=1, stride=1):
    '''
    You don't need to use this function directly.
    This is only defined for convenience to build class operations below.
    This function makes ONE filter scan the input tensor, and return the output.
    inp: input tensor, numpy array type
    filt: filter tensor, numpy array type
    bias: number
    '''
    # padding
    depth = inp.shape[0]
    width = inp.shape[1]
    new_width = width + 2 * padding
    new_input = np.zeros([depth, new_width, new_width])
    new_input[:, padding:new_width-padding, padding:new_width-padding] = inp

    # initialize output
    filt_size = filt.shape[1]
    output = np.zeros([(new_width-filt_size)//stride+1, (new_width-filt_size)//stride+1])

    # scanning
    for i in range(0,new_width-filt_size+1,stride):
        for j in range(0,new_width-filt_size+1,stride):
            scan_area = new_input[:, i:i+filt_size, j:j+filt_size]
            output[i//stride,j//stride] = np.sum(scan_area*filt) + bias

    return output

# Convlayer
class ConvLayer(Layer):
    '''
    ConvLayer is a subclass of Layer. It inherits all attributes of Layer.
    And it adds three attributes: filters, padding, stride
    'filters' should be a list, the element of which should be Filter type.
    '''
    def __init__(self, layer=np.array([[[0]]]), ltype='conv', filts=0, padding=1, stride=1, lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)
        self.filters = filts # list of filters and their biases
        self.padding = padding
        self.stride = stride

    def Conv(self):
        '''
        This is an operation of convolution in the forward process.
        We only need to use 'scan()' with every filter, and stack the output together.
        Notice that in the last, we set self.grad to zero.
        Because when a new image pass forward, the previous gradient (gradient from the last image) should be cleared.
        '''
        filts = self.filters
        padding = self.padding
        stride = self.stride
        bias = np.array([filts[j].bias for j in range(len(filts))])
        width = self.layer.shape[1]
        new_width = width + 2 * padding
        out_depth = len(filts)
        filt_size = filts[0].size
        output = np.zeros([out_depth, (new_width-filt_size)//stride+1, (new_width-filt_size)//stride+1])
        for i in range(out_depth):
            output[i,:,:] = scan(self.layer, filts[i].kernel, bias[i], padding=padding, stride=stride)
        # We did not use batch or mini-batch, so for each time a new image passes forward, the previous gradient is cleared.
        self.grad = np.zeros(self.grad.shape)
        for filt in self.filters:
            filt.kernelgrad = np.zeros(filt.kernelgrad.shape)
            filt.biasgrad = 0
        return output

    # Backpropagation
    def grad_Conv(self):
        '''
        This is the function of backpropagation of the conv layer.
        If you have any questions about the algorithm, you can refer to the slides 'CNN in Practice'.
        '''
        padding = self.padding
        stride = self.stride
        depth = self.depth
        width = self.width
        d_outp = self.nextlayer.grad
        filts = self.filters

        # padding
        new_width = width + 2 * padding
        new_input = np.zeros([depth, new_width, new_width]) # padded input
        new_input[:, padding:new_width-padding, padding:new_width-padding] = self.layer
        d_inp = np.zeros(new_input.shape)

        # scan and convolve
        for k in range(len(filts)):
            filt = self.filters[k]
            for i in range(0,new_width-filt.size+1,stride):
                for j in range(0,new_width-filt.size+1,stride):
                    scan_area = new_input[:, i:i+filt.size, j:j+filt.size]
                    d_inp[:, i:i+filt.size, j:j+filt.size] += d_outp[k, i//stride, j//stride] * filt.kernel
                    filt.kernelgrad += d_outp[k, i//stride, j//stride] * scan_area
                    filt.biasgrad += d_outp[k, i//stride, j//stride] * 1
        self.grad = d_inp[:, padding:new_width-padding, padding:new_width-padding]
        return d_inp[:, padding:new_width-padding, padding:new_width-padding]

    def update_Conv(self, step):
        '''
        This is the function to update the parameters of the filters and their biases.
        '''
        for filt in self.filters:
            #print(filt.kernel, filt.kernelgrad)
            filt.kernel -= filt.kernelgrad * step
            filt.bias -= filt.biasgrad * step

# Pool layer
class PoolLayer(Layer):
    '''
    PoolLayer is a subclass of Layer. It inherits all attributes of Layer.
    And it adds three attributes: filtersize, stride, poolindex
    We know that the process of pooling can be seen as a 'filter' scan over the input.
    (eg. 'filtsize=2' means that we find the maximal element from a 2*2 area.)
    'poolindex' records the index of the maximal elements, which helps in the backpropagation.
    '''
    def __init__(self, layer=np.array([[[0]]]), ltype='pool', filtsize=2, stride=2, lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)
        self.filtersize = filtsize
        self.stride = stride
        self.poolindex = 0

    # Max Pooling
    def Pool(self):
        '''
        This is an operation of max pooling.
        '''
        depth = self.depth
        width = self.width
        stride = self.stride
        filt_size = self.filtersize

        # initialize output
        output = np.zeros([depth, width//stride, width//stride])
        pool_index = np.zeros(output.shape)

        # scanning
        for d in range(depth):
            for i in range(0,width,stride):
                for j in range(0,width,stride):
                    scan_area = self.layer[d,i:i+filt_size, j:j+filt_size]
                    output[d,i//stride,j//stride] = np.max(scan_area)
                    pool_index[d,i//stride,j//stride] = np.argmax(scan_area)
        self.poolindex = pool_index
        # For each time a new image passes forward, the previous gradient is cleared.
        self.grad = np.zeros(self.grad.shape)
        return output

    def grad_Pool(self):
        '''
        This operation carries out backpropagation of max pooling.
        The local gradient of the maximal element of an area is 1,
        while others in this area is 0.
        '''
        depth = self.depth
        width = self.width
        stride = self.stride
        filt_size = self.filtersize
        d_outp = self.nextlayer.grad
        pool_index = self.poolindex
        for k in range(d_outp.shape[0]):
            for i in range(d_outp.shape[1]):
                for j in range(d_outp.shape[2]):
                    r = int(pool_index[k,i,j] // filt_size)
                    c = int(pool_index[k,i,j] % filt_size)
                    self.grad[k, (i*stride):(i*stride+filt_size), (j*stride):(j*stride+filt_size)][r,c] = 1 * d_outp[k,i,j]
        return self.grad

# ReLU Layer
class ReLULayer(Layer):
    '''
    ReLULayer is a subclass of Layer. It inherits all attributes of Layer.
    ReLULayer only carries out the operation max(x,0), when x is the output of the conv layer.
    Therefore, we don't need extra attributes.
    '''
    def __init__(self, layer=np.array([[[0]]]), ltype='relu', lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)

    # ReLU
    def ReLU(self):
        self.PorN = self.layer > np.zeros(self.layer.shape)
        return self.PorN * self.layer

    def grad_ReLU(self):
        d_outp = self.nextlayer.grad
        self.grad = self.PorN * d_outp
        return self.grad

# Softmax Classifier
class Classifier(Layer):
    '''
    SoftmaxLayer is a subclass of Layer. It inherits all attributes of Layer.
    It gives the probabilities of every class, based on the score given by the last layer.
    For more details about softmax, you can refer to the previous notes and slides, or Baidu.
    '''
    def __init__(self, layer=np.array([[[0]]]), ltype='softmax', lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)

    def Softmax(self, label):
        inp = self.layer
        inp -= np.max(inp)
        inp /= 20
        self.label = label
        return np.exp(inp[label])/np.sum(np.exp(inp))

    def grad_Softmax(self):
        label = self.label
        inp = self.layer
        self.grad = np.array([np.exp(inp[i])/np.sum(np.exp(inp)) - (i==label) for i in range(inp.shape[0])])/20
        return self.grad

class Network:
    '''
    This is a class of the whole network we build.
    You just need to input the structure you want, and the initial network can be built.
    eg. 'structure=[('conv',[1,28,28]), ('relu', [5,10,10])]' will build a network,
    of which the first layer is a 1*28*28 ConvLayer, and the second layer is a 5*10*10 ReLULayer.
    '''
    def __init__(self, structure):
        self.depth = len(structure)
        last = Layer()
        self.network = last
        for name, size in structure:
            if name=='conv':
                last.nextlayer = ConvLayer(layer=np.zeros(size), lastlayer = last)
            elif name=='relu':
                last.nextlayer = ReLULayer(layer=np.zeros(size), lastlayer = last)
            elif name=='pool':
                last.nextlayer = PoolLayer(layer=np.zeros(size), lastlayer = last)
            elif name=='softmax':
                last.nextlayer = Classifier(layer=np.zeros(size), lastlayer = last)
            last = last.nextlayer
        self.network = self.network.nextlayer

    def take_layer(self, index):
        '''
        This operation is to take 'index'th layer out.
        '''
        layer = self.network
        for i in range(index):
            layer = layer.nextlayer
        return layer

def Initialize(net, filts):
    '''
    We use Network() to build a network, but we do not attach any filters to it.
    This function attaches filters to the conv layers,
    and initializes all parameters with random numbers produces from normal distribution
    'net' is the network we build.
    (eg. filts=[(0,3), (3,5)], means attach filters of size 3 to the 0th layer,
    and filters of size 5 to the 3rd layer.)
    Notice we do not need to give the number of filters for each layer,
    because the network has been built, and the number of filters can be given by the depth of the next layer.
    '''
    for index, size in filts:
        this = net.take_layer(index)
        next = this.nextlayer
        this.filters = []
        for i in range(next.depth):
            filt = Filter(0.1*np.random.randn(this.depth, size, size), 0.1*np.random.randn())
            this.filters.append(filt)

def Train(net, dataset, labels, step, epochs):
    '''
    This function carries out the training process.
    'net': the network we have built and initialized
    'dataset': the training dataset without labels (X)
    'labels': the corresponding labels of dataset (Y)
    'step': update rate
    'epochs': eg. epochs=5 means iterate the whole dataset for 5 times
    '''
    step0 = step
    for epoch in range(epochs):
        for k in range(len(dataset)):
            
            # here we have a decay setting, you can change it if you like
            step = step0 /(k//1000+1)
            
            net.network.layer = dataset[k]
            this = net.network
            for i in range(net.depth):
                if this.layertype == 'conv':
                    next = this.Conv()
                elif this.layertype == 'relu':
                    next = this.ReLU()
                elif this.layertype == 'pool':
                    next = this.Pool()
                elif this.layertype == 'softmax':
                    next = this.Softmax(label=labels[k])
                    
                    # print the output of the softmax layer every 1000 steps
                    # you can change the setting here
                    loss = next
                    if k%1000==0:
                        print(k,' loss: ',loss, 'step ', step)
                    break
                
                this = this.nextlayer
                this.layer = next
            Backprop(net, step)
        # after iterating for an epoch, calculate the result on the training dataset
        accuracy = Test(net, dataset, labels)
    return accuracy

def Backprop(net, step):
    '''
    This is a function that backpropagate throughout the network
    'step': update rate
    '''
    this = net.take_layer(net.depth-1)
    for i in range(net.depth):
        if this.layertype == 'conv':
            this.grad_Conv()
            this.update_Conv(step)
        elif this.layertype == 'relu':
            this.grad_ReLU()
        elif this.layertype == 'pool':
            this.grad_Pool()
        elif this.layertype == 'softmax':
            this.grad_Softmax()
        this = this.lastlayer

def Test(net, dataset, labels):
    '''
    This is a function to test the dataset and give the accuracy.
    '''
    print('test on the dataset:')
    accuracy = 0
    for k in range(len(dataset)):
        net.network.layer = dataset[k]
        this = net.network
        for i in range(net.depth):
            if this.layertype == 'conv':
                next = this.Conv()
            elif this.layertype == 'relu':
                next = this.ReLU()
            elif this.layertype == 'pool':
                next = this.Pool()
            elif this.layertype == 'softmax':
                next = this.Softmax(label=labels[k])
                k0 = np.argmax(this.layer)
#                loss = next
#                if k%1000==0:
#                    print(k,' loss: ',loss, (labels[k],k0))
                if k0==labels[k]:
                    accuracy+=1
                break
            this = this.nextlayer
            this.layer = next
    accuracy /= len(dataset)
    print('accuracy:', accuracy, '\n')
    return accuracy

def Save(net, filename):
    '''
    This function is to save the parameters of the network to file.
    'filename' should be of str type that end with '.npz'.
    If you feel that you need more iterations for training,
    you can load the saved parameters, and don't need to train from beginning.
    '''
    layer = net.network
    para_list = []
    for i in range(net.depth):
        if layer.layertype == 'conv':
            filts = layer.filters
            kernel_list = []
            bias_list = []
            for filt in filts:
                kernel_list.append(filt.kernel)
                bias_list.append(filt.bias)
            para_list.append([kernel_list, bias_list])
        if layer.layertype == 'softmax':
            break
        layer = layer.nextlayer
    np.savez(filename, para_list)

def Load(net, filename):
    '''
    This function is to load the parameters you saved by 'Save()' function to your network
    '''
    parafile = np.load(filename)['arr_0']
    layer = net.network
    pos = 0
    for i in range(net.depth):
        if layer.layertype == 'conv':
            kernel_list = parafile[pos][0]
            bias_list = parafile[pos][1]
            for j in range(len(layer.filters)):
                filt = layer.filters[j]
                filt.kernel = kernel_list[j]
                filt.bias = bias_list[j]
            pos += 1
        if layer.layertype == 'softmax':
            break
        layer = layer.nextlayer
