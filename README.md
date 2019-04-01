# car-board-reg
基于CNN的车牌号识别
# 数据集介绍
## 车牌构成
为简化实验，在该实验中默认车牌字符已经得到划分，因此车牌识别可以分解为三个区域的字符识别任务（多分类任务），共实现7个字符的识别。
例如：`京A·F0236`
其中第一部分 `京` 表示车牌所在的省市,后面紧跟的`A`是发牌单位,间隔符`·`后面的5个字符就是序号。
省市Province：
("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新")
发牌单位Area：
("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z")
字符Letter:
("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z") 
## 数据集目录:
```
└─dataset
    ├─map.py    ---->映射关系文件
    ├─test      ---->测试集
    ├─train     ---->训练集
    │  ├─area  
    │  ├─letter
    │  └─province
    └─val       ----->验证集
        ├─area
        ├─letter
        └─province
```
验证集和训练集目录都各自包含3目录:area,letter,province.分别对应各自的含义。
```map
{
  "province": {
    "0": "皖",
    "1": "沪",
    "2": "津",
    "3": "渝",
    "4": "冀",
    "5": "晋",
    "6": "蒙",
    "7": "辽",
    "8": "吉",
    "9": "黑",
    "10": "苏",
    "11": "浙",
    "12": "京",
    "13": "闽",
    "14": "赣",
    "15": "鲁",
    "16": "豫",
    "17": "鄂",
    "18": "湘",
    "19": "粤",
    "20": "桂",
    "21": "琼",
    "22": "川",
    "23": "贵",
    "24": "云",
    "25": "藏",
    "26": "陕",
    "27": "甘",
    "28": "青",
    "29": "宁",
    "30": "新"
  },
  "area": {
    "0": "A",
    "1": "B",
    "2": "C",
    "3": "D",
    "4": "E",
    "5": "F",
    "6": "G",
    "7": "H",
    "8": "I",
    "9": "J",
    "10": "K",
    "11": "L",
    "12": "M",
    "13": "N",
    "14": "O",
    "15": "P",
    "16": "Q",
    "17": "R",
    "18": "S",
    "19": "T",
    "20": "U",
    "21": "V",
    "22": "W",
    "23": "X",
    "24": "Y",
    "25": "Z"
  },
  "letter": {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "A",
    "11": "B",
    "12": "C",
    "13": "D",
    "14": "E",
    "15": "F",
    "16": "G",
    "17": "H",
    "18": "J",
    "19": "K",
    "20": "L",
    "21": "M",
    "22": "N",
    "23": "P",
    "24": "Q",
    "25": "R",
    "26": "S",
    "27": "T",
    "28": "U",
    "29": "V",
    "30": "W",
    "31": "X",
    "32": "Y",
    "33": "Z"
  }
}
```
# PIL读取Image文件
本例提供的训练集里面的每个图片都是`20x20` 的二值化后的灰度图，例如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190401143457733.png)
因此，我们需要使用PIL库或opencv库把灰度图转换为我们方便处理的数据形式。本人是先转化为list of list.

picReader.py

```python
__author__ = 'jmh081701'
from PIL import  Image
def img2mat(img_filename):
#把所有的图片都resize为20x20
    img = Image.open(img_filename)
    img = img.resize((20,20))
    mat = [[img.getpixel((x,y)) for x in range(0,img.size[0])] for y in range(0,img.size[1])]
    return mat
def test():
    mat = img2mat("dataset\\test\\1.bmp")
    print(mat)
    print(mat[0][0],len(mat),len(mat[0]))
if __name__ == '__main__':
    test()
```
样例输出：

```shell
[[0, 0, 0, 0, 144, 212, 74, 17, 15, 60, 60, 62, 64, 67, 67, 68, 35, 0, 0, 0], [0, 0, 0, 0, 28, 119, 255, 101, 61, 233, 255, 255, 255, 255, 255, 255, 241, 44, 0, 0], [0, 0, 0, 0, 0, 15, 170, 92, 8, 14, 34, 31, 29, 29, 24, 74, 226, 38, 0, 0], [0, 0, 67, 220, 83, 4, 0, 0, 0, 84, 160, 0, 0, 0, 0, 52, 170, 11, 0, 0], [0, 0, 71, 255, 105, 10, 0, 0, 75, 230, 246, 124, 5, 0, 0, 49, 188, 19, 0, 0], [0, 0, 64, 255, 113, 15, 152, 216, 246, 255, 255, 255, 226, 225, 27, 46, 255, 59, 0, 0], [0, 0, 53, 255, 120, 22, 172, 249, 255, 255, 255, 255, 255, 255, 35, 33, 213, 61, 0, 0], [0, 0, 43, 255, 139, 105, 243, 254, 130, 231, 255, 139, 217, 255, 37, 35, 234, 63, 0, 0], [0, 0, 34, 247, 151, 68, 166, 248, 143, 225, 255, 159, 219, 255, 41, 37, 240, 50, 0, 0], [0, 0, 26, 240, 136, 38, 143, 246, 255, 255, 255, 255, 255, 255, 43, 29, 168, 0, 0, 0], [0, 0, 18, 231, 142, 44, 135, 246, 255, 255, 255, 230, 190, 98, 6, 25, 210, 49, 2, 0], [0, 0, 17, 223, 147, 49, 112, 214, 123, 226, 255, 147, 0, 0, 0, 28, 218, 10, 1, 0], [0, 0, 16, 212, 154, 56, 0, 0, 4, 69, 249, 149, 148, 216, 46, 18, 205, 94, 13, 0], [0, 0, 15, 200, 157, 59, 0, 11, 45, 255, 255, 242, 244, 255, 57, 3, 33, 13, 2, 0], [0, 0, 15, 196, 164, 66, 0, 66, 253, 198, 198, 198, 200, 225, 154, 87, 252, 90, 18, 0], [0, 0, 14, 184, 171, 73, 0, 8, 31, 1, 0, 0, 1, 16, 8, 13, 255, 110, 25, 0], [0, 0, 13, 175, 177, 79, 0, 0, 0, 0, 0, 0, 0, 0, 8, 37, 255, 117, 30, 0], [0, 0, 10, 134, 147, 69, 0, 0, 0, 0, 0, 0, 0, 0, 29, 127, 230, 24, 5, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 18, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
0 20 20
```
# 模型设计
数据集已经把一个车牌的三个部分都分开了，所以可以设计三个模型分别去识别这三部分。在本例中，本人为了简单期间，三个部分用了用一个CNN 网络结构，但是每个网络结构里面的具体参数是各自独立的。
CNN网络结构：
1. 输入层：20x20
2. 第一层卷积：卷积核大小：3x3，卷积核个数：32，Stride 步长：1，Same 卷积
3. 第二层卷积：卷积核大下：3x3，卷积核个数：64，Stride 步长：1，Same卷积
（两个卷积级联，效果就是5x5的卷积核，但是减少了参数个数）
4. 第二层池化：池化大小：2x2，max pool,Stride 步长：2
5. 第三层卷积：卷积核大小：3x3，卷积核个数：8，Stride 步长：1，Same卷积
6. 第三层池化：池化大小：2x2，max pooling,Stride :2。应该得到8个5x5的特征图。
平坦化：得到8x5x5=200维的向量
7. 第四层全连接：512个神经元，激活函数为relu。
8. 第五层全连接：34个神经元，softmax激活函数。

第五层是分类层，一共有34个神经元，表示最多有34个类别。对于province来说，只有前31类有效；对于area来说只有前26类有效；对于letter来说，这34个神经元都有效。因此在生成训练集的时候，需要把正确的答案标签编码为34维的one-hot
# 数据处理
数据处理模块，主要仿照minist数据集的写法，编写一个类，实现next_train_batch,next_test_batch,next_valid_batch函数。
BaseTool.py

```python
__author__ = 'jmh081701'
from dataset.map import  maps as aspectMap
import  os
from picReader import  img2mat
import  random
class data_generator:
    def __init__(self,aspect='area',seperate_ratio=0.1):
        '''
        :param aspect: 打开什么样的训练集,[area,letter,province] 三选一
        :param seperate_ratio: 测试集划分比例 ,从训练集和验证集里面随机抽取seperate_ratio作为训练集
        :return:
        '''
        self.train_dir = "dataset\\train\\%s\\" %aspect
        self.val_dir   = "dataset\\val\\%s\\" % aspect
        self.seperate_ratio = seperate_ratio

        self.data_vector_set  = []      #保存所有的图片向量
        self.data_label_set   = []      #保存所有的标签

        self.train_set = []             #保存训练集的下标
        self.train_batch_index = 0
        self.valid_set = []             #保存验证集的下标
        self.valid_batch_index = 0
        self.test_set = []              #保存测试集的下标
        self.test_batch_index = 0


        self.classes = 0                #最大的classes为34,这个值会在载入train和test后有所变化
        self.data_set_cnt = 0

        self.load_train()
        self.load_valid()

    def load_train(self):
        for rt,dirs,files in os.walk(self.train_dir):
            self.classes = max(self.classes,len(dirs))
            if len(dirs)==0 :
                #说明到了叶子目录,里面放着就是图片
                label = int(rt.split('\\')[-1])
                for name in files:
                    img_filename = os.path.join(rt,name)
                    vec = img2mat(img_filename)
                    self.data_vector_set.append(vec)
                    self.data_label_set.append(label)
                    if random.random() < self.seperate_ratio:
                        self.test_set.append(self.data_set_cnt)
                    else:
                        self.train_set.append(self.data_set_cnt)
                    self.data_set_cnt +=1
    def load_valid(self):
        for rt,dirs,files in os.walk(self.val_dir):
            self.classes = max(self.classes,len(dirs))
            if len(dirs)==0 :
                #说明到了叶子目录,里面放着就是图片
                label = int(rt.split('\\')[-1])
                for name in files:
                    img_filename = os.path.join(rt,name)
                    vec = img2mat(img_filename)
                    self.data_vector_set.append(vec)
                    self.data_label_set.append(label)
                    if random.random() < self.seperate_ratio:
                        self.test_set.append(self.data_set_cnt)
                    else:
                        self.valid_set.append(self.data_set_cnt)
                    self.data_set_cnt +=1
    def next_train_batch(self,batch=100):
        input_x =[]
        input_y =[]
        for i in range(batch):
            input_x.append(self.data_vector_set[self.train_set[(self.train_batch_index + i)%len(self.train_set)]])
            y = [0] * self.classes
            y[self.data_label_set[self.train_set[(self.train_batch_index +i)%len(self.train_set)]]] = 1
            input_y.append(y)
        self.train_batch_index +=batch
        self.train_batch_index %=len(self.train_set)
        return  input_x,input_y

    def next_valid_batch(self,batch=100):
        input_x =[]
        input_y =[]
        for i in range(batch):
            index = random.randint(0,len(self.valid_set)-1)
            input_x.append(self.data_vector_set[index])
            y = [0] * 34
            y[self.data_label_set[index]] = 1
            input_y.append(y)
        self.valid_batch_index +=batch

        self.valid_batch_index %=len(self.valid_set)
        return  input_x,input_y,self.train_epoch
    def next_test_batch(self,batch=100):
        input_x =[]
        input_y =[]
        for i in range(batch):
            input_x.append(self.data_vector_set[self.test_set[(self.test_batch_index + i)%len(self.test_set)]])
            y = [0] * self.classes
            y[self.data_label_set[self.test_set[(self.test_batch_index +i)%(len(self.test_set))]]] = 1
            input_y.append(y)
        self.test_batch_index +=batch
        self.test_batch_index %=len(self.test_set)
        return  input_x,input_y
if __name__ == '__main__':
    data_gen = data_generator()
    print(len(data_gen.test_set))
    print(data_gen.next_train_batch(50))
    print(data_gen.next_valid_batch(50)[1])
    print(data_gen.next_train_batch(30))
```

# 构建CNN模型
cnn_model.py

```python
__author__ = 'jmh081701'
import  tensorflow as tf
from BaseTool import  data_generator

batch_size = 100        # 每个batch的大小
learning_rate=1e-4      #学习速率
aspect = "area"
data_gen = data_generator(aspect)

input_x  =tf.placeholder(dtype=tf.float32,shape=[None,20,20],name='input_x')
input_y  =tf.placeholder(dtype=tf.float32,shape=[None,34],name='input_y')

with tf.name_scope('conv1'):
    W_C1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,32],stddev=0.1))
    b_C1 = tf.Variable(tf.constant(0.1,tf.float32,shape=[32]))

    X=tf.reshape(input_x,[-1,20,20,1])
    featureMap_C1 = tf.nn.relu(tf.nn.conv2d(X,W_C1,strides=[1,1,1,1],padding='SAME') + b_C1 )

with tf.name_scope('conv2'):
    W_C2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.1))
    b_C2 = tf.Variable(tf.constant(0.1,tf.float32,shape=[64]))
    featureMap_C2 = tf.nn.relu(tf.nn.conv2d(featureMap_C1,W_C2,strides=[1,1,1,1],padding='SAME') + b_C2)

with tf.name_scope('pooling2'):
    featureMap_S2 = tf.nn.max_pool(featureMap_C2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

with tf.name_scope('conv3'):
    W_C3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,8],stddev=0.1))
    b_C3 = tf.Variable(tf.constant(0.1,shape=[8],dtype=tf.float32))
    featureMap_C3 = tf.nn.relu(tf.nn.conv2d(featureMap_S2,filter=W_C3,strides=[1,1,1,1],padding='SAME')+ b_C3)

with tf.name_scope('pooling3'):
    featureMap_S3 = tf.nn.max_pool(featureMap_C3,[1,2,2,1],[1,2,2,1],padding='VALID')

with tf.name_scope('fulnet'):
    featureMap_flatten = tf.reshape(featureMap_S3,[-1,5*5*8])
    W_F4 = tf.Variable(tf.truncated_normal(shape=[5*5*8,512],stddev=0.1))
    b_F4 = tf.Variable(tf.constant(0.1,shape=[512],dtype=tf.float32))
    out_F4 = tf.nn.relu(tf.matmul(featureMap_flatten,W_F4) + b_F4)
    #out_F4 =tf.nn.dropout(out_F4,keep_prob=0.5)
with tf.name_scope('output'):
    W_OUTPUT = tf.Variable(tf.truncated_normal(shape=[512,34],stddev=0.1))
    b_OUTPUT = tf.Variable(tf.constant(0.1,shape=[34],dtype=tf.float32))
    logits = tf.matmul(out_F4,W_OUTPUT)+b_OUTPUT

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
predictY = tf.nn.softmax(logits)
y_pred=tf.arg_max(predictY,1)
bool_pred=tf.equal(tf.arg_max(input_y,1),y_pred)
right_rate=tf.reduce_mean(tf.to_float(bool_pred))

saver = tf.train.Saver()
def load_model(sess,dir,modelname):
    ckpt=tf.train.get_checkpoint_state(dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("*"*30)
        print("load lastest model......")
        saver.restore(sess,dir+".\\"+modelname)
        print("*"*30)
def save_model(sess,dir,modelname):
    saver.save(sess,dir+modelname)
dir = r".//"
modelname = aspect

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    display_interval=200
    max_epoch = 50
    epoch = 0
    acc = 0
    load_model(sess,dir=dir,modelname=modelname)
    while True  :
        if step % display_interval ==0:
            image_batch,label_batch,epoch = data_gen.next_valid_batch(batch_size)
            acc = sess.run(right_rate,feed_dict={input_x:image_batch,input_y:label_batch})
            print({'!'*30+str(epoch)+":"+str(step):acc})
        image_batch,label_batch,epoch = data_gen.next_train_batch(batch_size)
        sess.run([loss,train_op],{input_x:image_batch,input_y:label_batch})
        if(epoch> max_epoch):
            break
        step +=1
    while True :
        test_img,test_lab,test_epoch = data_gen.next_test_batch(batch_size)
        test_acc = sess.run(right_rate,{input_x:test_img,input_y:test_lab})
        acc = test_acc * 0.8 + acc * 0.2    #指数滑动平均
        if(test_epoch!=epoch):
            print({"Test Over..... acc:":acc})
            break
    save_model(sess,dir,modelname)

```
# 训练结果：
area：
```bash
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3:200': 0.34}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!6:400': 0.61}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!9:600': 0.78}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!13:800': 0.73}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!16:1000': 0.8}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!19:1200': 0.88}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!23:1400': 0.76}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!25:1600': 0.86}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!28:1800': 0.89}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!32:2000': 0.83}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!35:2200': 0.87}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!38:2400': 0.93}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!42:2600': 0.89}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!45:2800': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!48:3000': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!50:3200': 0.97}
{'Test Over..... acc:': 0.9042283506058594}
```
province:

```bash
******************************
load lastest model......
******************************
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!7:200': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!15:400': 0.88}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!22:600': 0.91}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!29:800': 0.92}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!36:1000': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!44:1200': 0.99}
{'Test Over..... acc:': 0.7786719818115235}
```
可以看到，模型有点过拟合了，因为验证集表现的很好，但是测试gg。
于是加上dropout

    out_F4 =tf.nn.dropout(out_F4,keep_prob=0.5)
结果，还是很差，哈哈哈，垃圾网络，233

```bash
******************************
load lastest model......
******************************
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!7:200': 0.91}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!15:400': 0.92}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!24:600': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!32:800': 0.92}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!40:1000': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!48:1200': 0.98}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!55:1400': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!63:1600': 0.96}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!71:1800': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!79:2000': 0.99}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!87:2200': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!95:2400': 0.98}
{'Test Over..... acc:': 0.857055978012085}
```
letter:
字符比较多，把max_epoch加大一些来。
```bash
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1:200': 0.02}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3:400': 0.06}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5:600': 0.07}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!6:800': 0.05}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!8:1000': 0.29}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!10:1200': 0.21}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11:1400': 0.34}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!13:1600': 0.41}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!15:1800': 0.51}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!16:2000': 0.48}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!18:2200': 0.51}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!20:2400': 0.68}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!21:2600': 0.48}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!23:2800': 0.64}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!24:3000': 0.76}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!26:3200': 0.65}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!27:3400': 0.64}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!29:3600': 0.71}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!31:3800': 0.77}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!32:4000': 0.75}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!34:4200': 0.74}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!36:4400': 0.82}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!37:4600': 0.8}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!39:4800': 0.77}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!41:5000': 0.82}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!42:5200': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!44:5400': 0.72}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!46:5600': 0.88}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!48:5800': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!48:6000': 0.58}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!50:6200': 0.85}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!52:6400': 0.91}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!53:6600': 0.85}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!55:6800': 0.89}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!57:7000': 0.91}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!58:7200': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!60:7400': 0.92}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!62:7600': 0.97}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!63:7800': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!65:8000': 0.85}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!67:8200': 0.91}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!69:8400': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!70:8600': 0.84}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!72:8800': 0.89}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!73:9000': 0.93}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!74:9200': 0.9}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!76:9400': 0.92}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!78:9600': 0.97}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!79:9800': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!81:10000': 0.96}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!83:10200': 0.96}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!84:10400': 0.93}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!86:10600': 0.81}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!88:10800': 0.97}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!90:11000': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!91:11200': 0.84}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!93:11400': 0.99}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!95:11600': 0.94}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!96:11800': 0.93}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!97:12000': 0.95}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!99:12200': 0.96}
{'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!100:12400': 0.98}
{'Test Over..... acc:': 0.8561504802688517}
```
