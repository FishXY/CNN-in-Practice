import numpy as np

# Data structure
class Filter:
    def __init__(self, filt, bias):
        self.kernel = filt
        self.bias = bias
        self.kernelgrad = np.zeros(filt.shape)
        self.biasgrad = 0
        if filt.shape != ():
            self.size = filt.shape[1]

class Layer:
    def __init__(self, layer=np.array([[[0]]]), ltype='layer', lastlayer=0, nextlayer=0):
        self.layer = layer
        self.lastlayer = lastlayer
        self.nextlayer = nextlayer
        self.grad = np.zeros(layer.shape)
        self.layertype = ltype
        self.depth = layer.shape[0]
        self.width = layer.shape[1]

def scan(inp, filt, bias, padding=1, stride=1):
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

#Convlayer
class ConvLayer(Layer):
    def __init__(self, layer=np.array([[[0]]]), ltype='conv', filts=0, padding=1, stride=1, lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)
        self.filters = filts # list of filters and their biases
        self.padding = padding
        self.stride = stride

    def Conv(self):
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
        for filt in self.filters:
            #print(filt.kernel, filt.kernelgrad)
            filt.kernel -= filt.kernelgrad * step
            filt.bias -= filt.biasgrad * step

#poollayer
class PoolLayer(Layer):
    def __init__(self, layer=np.array([[[0]]]), ltype='pool', filtsize=2, stride=2, lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)
        self.filtersize = filtsize
        self.stride = stride
        self.poolindex = 0

    # Max Pooling
    def Pool(self):
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

#ReLU
class ReLULayer(Layer):
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
    def __init__(self, layer=np.array([[[0]]]), ltype='softmax', lastlayer=0, nextlayer=0):
        Layer.__init__(self, layer, ltype, lastlayer, nextlayer)

    def Softmax(self, label):
        inp = self.layer
        inp -= np.max(inp)
        inp /= 20
        self.label = label
        #return -np.log((np.exp(inp[label])/np.sum(np.exp(inp))))
        return np.exp(inp[label])/np.sum(np.exp(inp))

    def grad_Softmax(self):
        label = self.label
        inp = self.layer
        self.grad = np.array([np.exp(inp[i])/np.sum(np.exp(inp)) - (i==label) for i in range(inp.shape[0])])/20
        return self.grad

# def Loss(softmaxout, reg=0):
#     global reg
#     return softmaxout

class Network:
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
        layer = self.network
        for i in range(index):
            layer = layer.nextlayer
        return layer

def Initialize(net, filts):
    for index, size in filts:
        this = net.take_layer(index)
        next = this.nextlayer
        this.filters = []
        for i in range(next.depth):
            filt = Filter(0.1*np.random.randn(this.depth, size, size), 0.1*np.random.randn())
            this.filters.append(filt)

def Train(net, dataset, labels, step, epochs):
    step0 = step
    for epoch in range(epochs):
#        print('epoch: ', epoch)
        for k in range(len(dataset)):
            step = step0 / (k//200 + 1)
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
                    loss = next
                    # if k%100==0:
                    #     print(k,' loss: ',loss, 'step ', step)
                    break
                this = this.nextlayer
                this.layer = next
            Backprop(net, step)
        accuracy = Test(net, dataset, labels)
    return accuracy

def Backprop(net, step):
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
    print('test on the training dataset:')
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
