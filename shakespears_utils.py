# Tải các gói
from __future__ import print_function  # Sử dụng tính năng print từ Python 3.x
from keras.callbacks import LambdaCallback  # Callback để thực hiện hàm ngẫu nhiên
from keras.models import Model, load_model, Sequential  # Các loại mô hình neural network
from keras.layers import Dense, Activation, Dropout, Input, Masking  # Các lớp layer
from keras.layers import LSTM  # LSTM layer
from keras.utils.data_utils import get_file  # Hàm để tải dữ liệu từ URL
from keras.preprocessing.sequence import pad_sequences  # Chuyển dữ liệu thành dạng chuỗi số
import numpy as np  # Thư viện đại số tuyến tính
import random  # Hàm tạo số ngẫu nhiên
import sys  # Thao tác với hệ thống
import io  # Thao tác với luồng dữ liệu

def build_data(text, Tx = 40, stride = 3):
    """
    Tạo tập huấn luyện bằng cách quét một cửa sổ kích thước Tx trên văn bản, với bước nhảy là 3.
    
    Arguments:
    text -- chuỗi, văn bản của bài thơ Shakespeare
    Tx -- độ dài chuỗi, số lần lặp (hoặc ký tự) trong một ví dụ huấn luyện
    stride -- cách mà cửa sổ dịch chuyển trong quá trình quét
    
    Returns:
    X -- danh sách các ví dụ huấn luyện
    Y -- danh sách các nhãn huấn luyện
    """
    
    X = []
    Y = []

    ### BẮT ĐẦU MÃ TẠI ĐÂY ### (≈ 3 dòng)
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])
    ### KẾT THÚC MÃ TẠI ĐÂY ###
    
    print('Số lượng ví dụ huấn luyện:', len(X))
    
    return X, Y


def vectorization(X, Y, n_x, char_indices, Tx = 40):
    """
    Chuyển đổi X và Y (danh sách) thành các mảng để đưa vào mạng neural hồi tiếp.
    
    Arguments:
    X -- 
    Y -- 
    Tx -- số nguyên, độ dài chuỗi
    
    Returns:
    x -- mảng có hình dạng (m, Tx, len(chars))
    y -- mảng có hình dạng (m, len(chars))
    """
    
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.bool)
    y = np.zeros((m, n_x), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[Y[i]]] = 1
        
    return x, y 


def sample(preds, temperature=1.0):
    # Hàm hỗ trợ để lấy một chỉ số từ một mảng xác suất
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p = probas.ravel())
    return out
    #return np.argmax(probas)
    
def on_epoch_end(epoch, logs):
    # Hàm được gọi sau mỗi epoch. In ra văn bản được tạo ra.
    None
    #start_index = random.randint(0, len(text) - Tx - 1)
    
    #generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    #usr_input = input("Write the beginning of your poem, the Shakespearian machine will complete it.")
    # zero pad the sentence to Tx characters.
    #sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    #generated += sentence
#
    #sys.stdout.write(usr_input)

    #for i in range(400):
"""
        #x_pred = np.zeros((1, Tx, len(chars)))

        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature = 1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
        
        if next_char == '\n':
            continue
        
    # Dừng ở cuối một dòng (4 dòng)
    print()
 """   
print("Đang tải dữ liệu văn bản...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
#print('corpus length:', len(text))

Tx = 40
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
#print('number of unique characters in the corpus:', len(chars))

print("Tạo tập huấn luyện...")
X, Y = build_data(text, Tx, stride = 3)
print("Chuyển đổi tập huấn luyện thành dạng số...")
x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices) 
print("Đang tải mô hình...")
model = load_model('models/model_shakespeare_kiank_350_epoch.h5')


def generate_output():
    generated = ''
    #sentence = text[start_index: start_index + Tx]
    #sentence = '0'*Tx
    usr_input = input("Viết đoạn bài thơ của bạn, máy Shakespeare sẽ hoàn thiện nó. Đầu vào của bạn là: ")
    # zero pad the sentence to Tx characters.
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower
