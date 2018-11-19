import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def layer(Inp, Inp_size, Neuron_num, lay_name, keep_prob=1, Act_function = None):
    with tf.name_scope(str(lay_name)):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([Inp_size, Neuron_num], dtype=tf.float64), name='Weight')
            tf.summary.histogram('Weight', Weight)
        with tf.name_scope('Basic'):
            Basic = tf.Variable(tf.random_normal([1, Neuron_num], dtype=tf.float64), name='Basic')
            tf.summary.histogram('Basic', Basic)
        with tf.name_scope('Out_Put'):
            # Outputs_temp = tf.add(tf.matmul(Inp, Weight), Basic)
            Outputs_temp = tf.nn.dropout(tf.add(tf.matmul(Inp, Weight),Basic), keep_prob)
            if Act_function is None:
                Outputs = Outputs_temp
            else:
                Outputs = Act_function(Outputs_temp)
            tf.summary.histogram('Output', Outputs)

    return Outputs


def loss_step(Y_data, Y_pre):
    with tf.name_scope('Loss'):
        loss_pre = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(Y_data-Y_pre), reduction_indices=[1])))
        tf.summary.scalar('Loss', loss_pre)
    return loss_pre


def train_step(loss):
    with tf.name_scope('Train'):
        train = tf.train.MomentumOptimizer(0.0000001,0.9).minimize(loss)
    return train


def sess_step(mode=3, proportion=0.333):
    if mode == 1:
        # 训练方式为指定使用一定比例的Gpu容量
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=proportion)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    elif mode == 2:
        # 训练方式为按使用增长所占Gpu容量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        # 使用cpu训练模型
        sess = tf.Session()
    return sess
# # 计算相关矩阵
# def xcorr(a, b=None):
#     if b is None:
#         b = a
#     elif len(a) > len(b):
#         c = np.zeros(len(a))
#         for index in range(len(b)):
#             c[index] = b[index]
#         b = c
#     elif len(a) < len(b):
#         c = np.zeros(len(b))
#         for index in range(len(a)):
#             c[index] = a[index]
#         a = b
#         b = c
#     else :
#         pass
#     temp = np.zeros(len(b)*3-2)
#     temp[len(b)-1:2*len(b)-1] = b
#     corr_mat = np.zeros((len(a),len(a)))
#     for i in range(len(a)):
#         for j in range(len(a)):
#             corr_mat[i][j] = np.correlate(a, temp[len(a)-1+i-j:2*len(a)-1+i-j])
#     print(corr_mat)

# 读入数据
Distance = np.loadtxt('G:\ANN\Distance.txt', delimiter='\n')[:, np.newaxis]*1000
Wavelength = np.loadtxt('G:\ANN\Reference_wavelengths.txt', delimiter='\n')
Ref_sp = np.loadtxt('G:\ANN\Reference_spectrums.txt', delimiter='\n')
Raw = 0
for index_fir, line_fir in enumerate(open('G:\ANN\Raw_spectrumArray.txt', 'r')):
    Raw += 1
Raw_sp = np.zeros((Raw, Wavelength.shape[0]))
for index_sec, line_sec in enumerate(open('G:\ANN\Raw_spectrumArray.txt')):
    temp = line_sec.split('       ')
    temp.pop()
    Raw_sp[index_sec] = temp
# 测量值与参考数值做差
Raw_sp = Raw_sp-Ref_sp
# 对数据进行滤波
# ALPHA 滤波
Template_size = 25
Drop_number = 12
for i_index in range(Raw_sp.shape[0]):
    temp = Raw_sp[i_index]
    for j_index in range(Raw_sp.shape[1]):
        if j_index >= int((Template_size-1)/2):
            temp_section = temp[j_index - int((Template_size - 1) / 2):j_index + int((Template_size - 1) / 2) + 1]
            temp_section_dropped = np.sort(temp_section)[int(Drop_number/2):Template_size-int(Drop_number/2)]
            Raw_sp[i_index,j_index] = np.mean(temp_section_dropped)

# 归一化
for index in range(Raw_sp.shape[0]):
    Raw_sp[index] = Raw_sp[index]/Raw_sp[index].max()
# Raw_sp = np.where(Raw_sp > 0.45, Raw_sp, 0)

# 提取数据
Raw_sp_pro = np.zeros((Raw_sp.shape[0],41))
for index in range(Raw_sp.shape[0]):
    loc = np.where(Raw_sp[index]==Raw_sp[index].max())[0][0]
    Raw_sp_pro[index,0:40]=Raw_sp[index,loc-20:loc+20]
    Raw_sp_pro[index,40]=Wavelength[loc]
Raw_sp_pro[:,40] = (Raw_sp_pro[:,40]-Raw_sp_pro[:,40].min())/(Raw_sp_pro[:,40].max()-Raw_sp_pro[:,40].min())

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(Raw_sp_pro, Distance, test_size=0.05, random_state=42)
# 构造batch
with tf.name_scope('Input_Data'):
    Xs = tf.placeholder(tf.float64, [None, X_train.shape[1]-1], name='Xs')
    Ys = tf.placeholder(tf.float64, [None, Y_train.shape[1]], name='Ys')
    Zs = tf.placeholder(tf.float64, [None, 1], name='Zs')
    keep_prob = tf.placeholder(tf.float64)
# 构造第一个神经网络，初始化损失函数与训练
L1 = layer(Xs, 40, 70, lay_name='L_One', Act_function=tf.nn.sigmoid)
L2 = layer(L1, 70, 50, lay_name='L_Two', Act_function=tf.nn.relu)
L3 = layer(L2, 50, 30, lay_name='L_Three', Act_function=tf.nn.sigmoid)
L4 = layer(L3, 30, 10, lay_name='L_Four', Act_function=tf.nn.relu)
L5 = layer(L4, 10, 10, lay_name='L_Five', Act_function=tf.nn.sigmoid)

L6 = layer(L5, 10, 70, lay_name='L_Six', Act_function=tf.nn.relu)
L7 = layer(L6, 70, 50, lay_name='L_Seven', Act_function=tf.nn.sigmoid)
L8 = layer(L7, 50, 30, lay_name='L_Eight', Act_function=tf.nn.relu)
L9 = layer(L8, 30, 10, lay_name='L_Nine', Act_function=tf.nn.sigmoid)
L10 = layer(L9, 10, 10, lay_name='L_Ten', Act_function=tf.nn.relu)

L11 = layer(Zs, 1, 70, lay_name='L_Eleven', Act_function=tf.nn.sigmoid)
L12 = layer(L11, 70, 50, lay_name='L_Twelve', Act_function=tf.nn.sigmoid)
L13 = layer(L12, 50, 30, lay_name='L_Thirteen', Act_function=tf.nn.relu)
L14 = layer(L13, 30, 10, lay_name='L_Fourteen', Act_function=tf.nn.sigmoid)
L15 = layer(L14, 10, 10, lay_name='L_Fifteen', Act_function=tf.nn.sigmoid)


L16 = layer(L15+L15, 10, 70, lay_name='L_Sixteen', Act_function=tf.nn.sigmoid)
L17 = layer(L16, 70, 50, lay_name='L_Seventeen',Act_function=tf.nn.relu)
L18 = layer(L17, 50, 30, lay_name='L_Eighteen',Act_function=tf.nn.relu)
L19 = layer(L18, 30, 10, lay_name='L_Nineteen',Act_function=tf.nn.relu)
Y_pre = layer(L19, 10, 1, lay_name='L_Out',Act_function=tf.nn.relu)

loss = loss_step(Ys, Y_pre)
train = train_step(loss)
with sess_step() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test', sess.graph)
    just = 0
    loss_train = 100
    contrast_train_loss = 1
    contrast_test_loss = 1
    fig = plt.figure(num='Lz_Studio',figsize=(10,5),dpi=400)
    while loss_train > 25:
        sess.run(train, feed_dict={Xs: X_train[:,0:40], Ys: Y_train,Zs:X_train[:,40][:,np.newaxis], keep_prob:1})
        just += 1
        if just % 500 == 0:
            loss_train = sess.run(loss, feed_dict={Xs: X_train[:,0:40], Ys: Y_train,Zs:X_train[:,40][:,np.newaxis], keep_prob:1})
            loss_test = sess.run(loss, feed_dict={Xs: X_test[:,0:40], Ys: Y_test,Zs:X_test[:,40][:,np.newaxis], keep_prob:1})
            train_result = sess.run(merged, feed_dict={Xs: X_train[:,0:40], Ys: Y_train,Zs:X_train[:,40][:,np.newaxis], keep_prob:1})
            test_result = sess.run(merged, feed_dict={Xs: X_train[:,0:40], Ys: Y_train,Zs:X_train[:,40][:,np.newaxis], keep_prob:1})
            contrast_train_loss = np.max(np.abs(sess.run(Y_pre, feed_dict={Xs: X_train[:,0:40],Zs:X_train[:,40][:,np.newaxis], keep_prob:1}) - Y_train))
            contrast_test_loss = np.max(np.abs(sess.run(Y_pre, feed_dict={Xs: X_test[:,0:40],Zs:X_test[:,40][:,np.newaxis], keep_prob:1}) - Y_test))
            train_writer.add_summary(train_result, just)
            test_writer.add_summary(test_result, just)
            print('number', just, ':', loss_train ,'(nm)', loss_test, '(nm)', 'contrast_loss:',
                  contrast_train_loss, '(nm)', contrast_test_loss, '(nm)')
        if just % 1000 == 0:
            # 获得预测值与真实值矩阵
            # 训练集矩阵[0]为训练预测数据标签[1]为训练真实数据标签
            Train_result = np.zeros((2,X_train.shape[0]))
            Train_result[0] = sess.run(Y_pre, feed_dict={Xs: X_train[:,0:40],Zs:X_train[:,40][:,np.newaxis], keep_prob:1}).reshape((1,X_train.shape[0]))
            Train_result[1] = Y_train.reshape((1,X_train.shape[0]))
            # 测试集矩阵[0]为测试预测数据标签[1]为测试真实数据标签
            Test_result = np.zeros((2,X_test.shape[0]))
            Test_result[0] = sess.run(Y_pre, feed_dict={Xs: X_test[:,0:40],Zs:X_test[:,40][:,np.newaxis], keep_prob:1}).reshape((1,X_test.shape[0]))
            Test_result[1] = Y_test.reshape((1,X_test.shape[0]))

            Train = Train_result.T[np.lexsort(Train_result)].T
            Test = Test_result.T[np.lexsort(Test_result)].T
            np.save('train_result.npy', Train)
            np.save('test_result.npy', Test)

            # 绘制误差图与位置对比图
            plt.clf()
            plt.subplots_adjust(hspace=0.5,wspace=0.3)
            ax_train_0 = plt.subplot(221)
            ax_train_0.plot(Train[1],(Train[1]-Train[0]),'r')
            ax_train_0.set_title('Train Process')
            ax_train_0.set_xlabel('Real Position(/nm)')
            ax_train_0.set_ylabel('Error(/nm)')

            ax_test_0 = plt.subplot(222)
            ax_test_0.plot(Test[1],(Test[1]-Test[0]),'r')
            ax_test_0.set_title('Test Process')
            ax_test_0.set_xlabel('Real Position(/nm)')
            ax_test_0.set_ylabel('Error(/nm)')

            ax_train_1 = plt.subplot(223)
            ax_train_1.plot(Train[1], Train[1], 'r',label='Real Position')
            ax_train_1.plot(Train[1], Train[0], 'b',label='Pred Position')
            ax_train_1.set_title('Train Data Contrast')
            ax_train_1.set_xlabel('Real Position(/nm)')
            ax_train_1.set_ylabel('Position(/nm)')
            ax_train_1.legend()

            ax_test_1 = plt.subplot(224)
            ax_test_1.plot(Test[1], Test[1], 'r-',label='Real Position')
            ax_test_1.plot(Test[1], Test[0], 'b-',label='Pred Position')
            ax_test_1.set_title('Test Data Contrast')
            ax_test_1.set_xlabel('Real Position(/nm)')
            ax_test_1.set_ylabel('Position(/nm)')
            ax_test_1.legend()

            plt.savefig('Result.png')
