__author__ = 'jmh081701'
import  tensorflow as tf
from BaseTool import  data_generator

batch_size = 100        # 每个batch的大小
learning_rate=1e-4      #学习速率
aspect = "letter"
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
    out_F4 =tf.nn.dropout(out_F4,keep_prob=0.5)
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
dir = r".//parameter//%s//"%aspect
modelname = aspect

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    display_interval=200
    max_epoch = 100
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
