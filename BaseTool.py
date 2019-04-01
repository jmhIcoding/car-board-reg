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
            input_x.append(self.data_vector_set[self.valid_set[(self.valid_batch_index + i)%len(self.valid_set)]])
            y = [0] * self.classes
            y[self.data_label_set[self.valid_set[(self.valid_batch_index +i)%(len(self.valid_set))]]] = 1
            input_y.append(y)
        self.valid_batch_index +=batch
        self.valid_batch_index %=len(self.valid_set)
        return  input_x,input_y
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