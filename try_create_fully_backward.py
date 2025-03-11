import numpy as np
from matmul_gpu import matmulGPU # dùng opencl hổ trợ gpu tích hợp

class ActivationFunction:
    def __init__(self):
        pass

    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def relu(self, x):
        return np.maximum(x, 0)

class BatchNorm3D:
    def __init__(self, features, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones((1, 1, features))  # Hệ số scale
        self.beta = np.zeros((1, 1, features))  # Hệ số dịch chuyển
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros((1, 1, features))
        self.running_var = np.ones((1, 1, features))

    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=(0, 1), keepdims=True)
            var = np.var(x, axis=(0, 1), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class LossFunction:
    def __init__(self):
        pass

    def sparse_categorical_crossentropy(self, y_pred, y_true):
        dL = y_pred.copy()
        L = y_pred.copy()
        errors = []
        batch_size, seq_len = y_true.shape
        for i in range(batch_size):
            for j in range(seq_len):
                errors.append(L[i, j, y_true[i, j]])
                dL[i, j, y_true[i, j]] -= 1
        errors = np.stack(errors)
        return np.mean(-np.log(errors)), dL
    
    def categorical_crossentropy(self, y_pred, y_true):
        dL = y_pred.copy()
        L = y_pred.copy()
        errors = []
        batch_size = y_true.shape[0]
        for i in range(batch_size):
            errors.append(L[i, y_true[i, 0]])
            dL[i, y_true[i, 0]] -= 1
        errors = np.stack(errors)
        return np.mean(-np.log(errors)), dL
    
class LayersSequence:
    def __init__(self, layers: list, loss_fn=LossFunction):
        self.layers = layers
        self.loss_fn = loss_fn()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward_sparse_categorical_crossentropy(self, y_pred, y_true, lr=0.0001):
        L, dL = self.loss_fn.sparse_categorical_crossentropy(y_pred, y_true)
        for layer in reversed(self.layers):
            dL = layer.backward_sparse_categorical_crossentropy(dL, lr)
        return L
    
    def backward_categorical_crossentropy(self, y_pred, y_true, lr=0.0001):
        L, dL = self.loss_fn.sparse_categorical_crossentropy(y_pred, y_true)
        for layer in reversed(self.layers):
            dL = layer.backward_categorical_crossentropy(dL, lr)
        return L
    
class Linear:
    def __init__(self, in_features, out_features, activation="softmax", normalization=BatchNorm3D):
        self.weight = np.random.rand(in_features, out_features) * np.sqrt(1.0 / in_features)
        self.bias = np.random.rand(out_features) * np.sqrt(1.0 / in_features)
        self.activation_fn = ActivationFunction()

        if normalization is not None:
            self.norm = normalization(out_features)
        else:
            self.norm = None

        self.activation = activation
        self.output = None
        self.input = None
        self.batch_size = None
        self.seq_len = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.seq_len = x.shape[1]
        self.input = x.copy()
        weight_out = matmulGPU(x, self.weight)
        
        if self.activation == "softmax":
            self.output = self.activation_fn.softmax(weight_out + self.bias)
        elif self.activation == "relu":
            self.output = self.activation_fn.relu(weight_out + self.bias)
        else:
            self.output = weight_out + self.bias
        
        if self.activation != "softmax" and self.norm is not None:
            return self.norm.forward(self.output)
        else:
            return self.output
    
    def backward_sparse_categorical_crossentropy(self, dL, lr=0.0001):
        nn_input = self.input.copy()
        nn_output = self.output.copy()
        nn_weight = self.weight.copy()
        nn_dL = dL.copy()
        
        dW = matmulGPU(nn_input.transpose(0, 2, 1), nn_dL)
        dB = np.sum(np.sum(dL, axis=0))
        
        # cập nhật trọng số theo hàm loss sparse_categorical_crossentropy
        for i in range(self.batch_size):
            self.weight -= lr * dW[i, :, :]
        self.bias -= lr * dB
        
        # kiểm soát giá trị trọng số tránh tràn số, tối ưu tốc độ học, ổn định gradient
        while np.max(self.weight) > 2.0 or np.max(self.bias) > 2.0 or np.min(self.weight) < -2.0 or np.min(self.bias) < -2.0:
            if np.max(self.weight) > 2.0:
                self.weight -= (self.weight - 1.0)
            elif np.max(self.bias) > 2.0:
                self.bias -= (self.bias - 1.0)
            elif np.min(self.weight) < -2.0:
                self.weight -= (self.weight + 1.0)
            elif np.min(self.bias) < -2.0:
                self.bias -= (self.bias + 1.0)
        
        output_error = matmulGPU(dL, nn_weight.T)
        return output_error
    
    def backward_categorical_crossentropy(self, dL, lr=0.0001):
        nn_input = self.input.copy()
        nn_output = self.output.copy()
        nn_weight = self.weight.copy()
        nn_dL = dL.copy()
        dW = matmulGPU(nn_input.transpose(0, 2, 1), nn_dL)
        dB = np.sum(dL, axis=0)
        output_error = matmulGPU(dL, nn_weight.T)
    
        self.weight -= lr * dW
        self.bias -= lr * dB

        return output_error
    
class MultiheadSelfAttention:
    def __init__(self, d_model, num_head, ffn_dim):
        self.qkv_layer = Linear(
            in_features=d_model,
            out_features=d_model*3,
            activation=None,
            normalization=None
        )
        self.ffn_layer_1 = Linear(
            in_features=d_model,
            out_features=ffn_dim,
            activation="relu",
            normalization=None
        )
        self.ffn_layer_2 = Linear(
            in_features=ffn_dim,
            out_features=d_model,
            activation="relu",
            normalization=None
        )

        self.activation = ActivationFunction()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.head_dim = d_model // num_head
        self.num_head = num_head
        self.d_model = d_model

        # history data for grad
        self.attention_scores = None
        self.Q = None
        self.K = None
        self.V = None

    def att_block_forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_layer.forward(x)
        qkv = qkv.reshape(batch_size, self.num_head, seq_len, 3*self.head_dim)
        Q, K, V = (
            qkv[:, :, :, :self.head_dim],
            qkv[:, :, :, self.head_dim:self.head_dim*2],
            qkv[:, :, :, self.head_dim*2:self.head_dim*3]
        )
        d_k = np.sqrt(Q.shape[-1])
        scaled = matmulGPU(Q, K.transpose(0, 1, 3, 2)) / d_k
        if mask is not None:
            scaled += mask
        attention_scores = self.activation.softmax(scaled)
        values = matmulGPU(attention_scores, V)

        self.attention_scores = attention_scores
        self.Q = Q
        self.K = K
        self.V = V
        
        values = values.reshape(batch_size, seq_len, self.d_model)
        return attention_scores, values
    
    def ffn_block_forward(self, x):
        x = self.ffn_layer_1.forward(x)
        x = self.ffn_layer_2.forward(x)
        return x
    
    def att_block_backward(self, dL, lr=0.0001):
        output_error = dL.copy()
        batch_size, seq_len = output_error.shape[0], output_error.shape[1]
        output_error = output_error.reshape(
            batch_size, self.num_head, seq_len, self.head_dim
        )
        dA = matmulGPU(output_error, self.V.transpose(0, 1, 3, 2))
        dV = matmulGPU(self.attention_scores.transpose(0, 1, 3, 2), output_error)
        dK = matmulGPU(dA.transpose(0, 1, 3, 2), self.Q)
        dQ = matmulGPU(dA.transpose(0, 1, 3, 2), self.K)

        dQKV = np.concatenate([dQ, dK, dV], axis=-1)
        dQKV = dQKV.reshape(batch_size, seq_len, 3*self.d_model)
        dL = self.qkv_layer.backward_sparse_categorical_crossentropy(dQKV, lr)
        return dL

    def ffn_block_backward(self, dL, lr=0.0001):
        dL = self.ffn_layer_2.backward_sparse_categorical_crossentropy(dL, lr)
        dL = self.ffn_layer_1.backward_sparse_categorical_crossentropy(dL, lr)
        return dL

    def forward(self, x):
        pre_x = x.copy()
        att_scores, x = self.att_block_forward(x, None)
        x = self.norm1.forward(x + pre_x)
        pre_x = x.copy()
        x = self.ffn_block_forward(x)
        x = self.norm2.forward(x + pre_x)
        return x
    
    def backward_sparse_categorical_crossentropy(self, dL, lr=0.0001):
        dL = self.ffn_block_backward(dL, lr)
        dL = self.att_block_backward(dL, lr)
        return dL
    
model = LayersSequence(
    [
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        MultiheadSelfAttention(d_model=512,num_head=16,ffn_dim=1024),
        Linear(512, 26, activation="softmax", normalization=None)
    ]
)

loss_fn = LossFunction()
embed = np.random.rand(26, 512) * np.sqrt(1.0 / 512)
x = np.random.randint(0, 26, (5, 20))
y_true = np.random.randint(0, 25, (5, 20))

for epoch in range(500):
    y_pred = model.forward(embed[x])
    loss = model.backward_sparse_categorical_crossentropy(y_pred, y_true, 0.001)
    print(f"Epoch {epoch+1} with Loss: {loss}")


y_pred = model.forward(embed[x[:1, :4]])
print("kết quả thực tế:\n",y_true[0, :4])
print("kết quả dự đoán:\n",np.argmax(y_pred, -1))