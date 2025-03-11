import numpy as np

def softmax(x):
    # Xử lý tràn số học (numerical stability) bằng cách trừ đi giá trị max của x từ mỗi phần tử
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # Tránh overflow
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

X = np.random.rand(32, 50, 512)
y_true = np.random.randint(low=0, high=25, size=(32, 50))

# khởi tạo trọng số
wI = np.random.randn(512, 128) * np.sqrt(2.0 / 512)  # He init
wH = np.random.randn(128, 256) * np.sqrt(2.0 / 128)
wO = np.random.randn(256, 25) * np.sqrt(2.0 / 256)

# Lan truyền tuyến test
# oI = (X @ wI)
# oH = (oI @ wH)
# y_pred = softmax(oH @ wO)

# print("Y dự đoán ban đầu:", np.argmax(y_pred, axis=-1))
# print("Y thực tế:", y_true)

for epoch in range(1000):
    # Lan truyền tuyến
    oI = (X @ wI)
    oH = (oI @ wH)
    y_pred = softmax(oH @ wO)

    # Tính toán lỗi phân loại đa lớp cho sequence (sparse categorical crossentropy)
    batch_size, seq_len = y_true.shape
    dL = y_pred.copy()
    for i in range(batch_size):
        for j in range(seq_len):
            dL[i, j, y_true[i, j]] -= 1

    # tính toán lan truyền ngược, dựa trên lỗi của sparse categorical crossentropy
    output_error = dL.copy()
    hidden_error = output_error @ wO.T
    input_error = hidden_error @ wH.T

    dWO = np.sum(
        np.matmul(oH.transpose(0, 2, 1), output_error),
        axis=0
    )
    dWH = np.sum(
        np.matmul(oI.transpose(0, 2, 1), hidden_error),
        axis=0
    )
    dWI = np.sum(
        np.matmul(X.transpose(0, 2, 1), input_error),
        axis=0
    )

    dWO /= batch_size
    dWH /= batch_size
    dWI /= batch_size

    # print("dWO:", dWO.shape)
    # print("dWH:", dWH.shape)
    # print("dWI:", dWI.shape)

    wO -= 0.0001 * dWO
    wH -= 0.0001 * dWH
    wI -= 0.0001 * dWI
    
    probs = []
    for i in range(batch_size):
        for j in range(seq_len):
            probs.append(y_pred[i, j, y_true[i, j]])
    probs = np.stack(probs)

    L = np.mean(-np.log(probs))
    print("Epoch:", epoch, "Loss:", L)

# Lan truyền tuyến test
# oI = (X @ wI)
# oH = (oI @ wH)
# y_pred = softmax(oH @ wO)

# print("Y dự đoán:", np.argmax(y_pred, axis=-1))
# print("Y thực tế:", y_true)