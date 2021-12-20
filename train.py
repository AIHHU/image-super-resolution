#!/usr/bin/env python
# coding: utf-8



#!tar -xf /home/aistudio/data/data110995/DIV2K.tar




import os
import shutil
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from visualdl import LogWriter
paddle.enable_static()





writer=LogWriter(logdir='work/log/')


def conv_bn_layer(input,num_filters,filter_size,stride=1,padding="SAME",act=None):
    conv=fluid.layers.conv2d(input=input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            act=None)
    return fluid.layers.batch_norm(input=conv,act=act)
image=fluid.data(name='image',shape=[None,3,None,None],dtype='float32')
image_bigger=paddle.nn.functional.upsample(image,scale_factor=(4,4),mode="BICUBIC")
X=conv_bn_layer(image_bigger,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=conv_bn_layer(X,64,(3,3),act="relu")
X=fluid.layers.conv2d(input=X,num_filters=1,filter_size=(3,3),stride=1,padding="SAME",act=None)
result=fluid.layers.elementwise_add(x=image_bigger,y=X,act="relu")
label=fluid.data(name='label',shape=[None,3,None,None],dtype='float32')
cost=fluid.layers.square_error_cost(input=result, label=label)
avg_cost=fluid.layers.mean(cost)
test_program=fluid.default_main_program().clone(for_test=True)
optimizer=fluid.optimizer.RMSProp(learning_rate=0.001)
opts=optimizer.minimize(avg_cost)




import numpy as np
from PIL import Image
x_base="DIV2K/DIV2K_train_LR_bicubic/X4/"
y_base="DIV2K/DIV2K_train_HR/"
x_dirs=os.listdir(x_base)
y_dirs=os.listdir(y_base)
y_dirs.sort()
x_dirs.sort()
def load_image(dirname):
    a=Image.open(dirname)
    a=np.array(a).astype("float32").transpose([2,0,1])/255
    a=a.reshape(1,a.shape[0],a.shape[1],a.shape[2])
    return a

test_x=[]
test_y=[]
j=800
while j<900:
    test_x.append(load_image(x_base+x_dirs[j]))
    test_y.append(load_image(y_base+y_dirs[j]))
    j=j+1




place=fluid.CUDAPlace(0)
exe=fluid.Executor(place)
exe.run(fluid.default_startup_program())




train_step=0
test_step=0
min_cost=1000
save_path='work/best_model/'
patience=0
for pass_id in range(20):
    x_data=[]
    y_data=[]
    if pass_id%2==0:
        i=0
    else:
        i=400
    end=i+400
    while i<end:
        x_data.append(load_image(x_base+x_dirs[i]))
        y_data.append(load_image(y_base+y_dirs[i]))
        i=i+1
    for batch_id,data in enumerate(x_data):
        train_cost=exe.run(program=fluid.default_main_program(),
                                    feed={image.name:data, label.name:y_data[batch_id]},
                                    fetch_list=[avg_cost])
        train_step +=1
        writer.add_scalar(tag="训练/损失值",step=train_step,value=train_cost[0])
    print("Pass:%d, Batch:%d, Cost:%0.5f" % (pass_id, batch_id,train_cost[0]))
    if pass_id%2!=0:
        test_costs=[]
        for batch_id,data in enumerate(test_x):
            test_cost=exe.run(program=test_program,
                            feed={image.name:data, label.name:test_y[batch_id]},
                            fetch_list=[avg_cost])
            test_step +=1
            writer.add_scalar(tag="测试/损失值",step=train_step,value=test_cost[0])
            test_costs.append(test_cost[0])
        test_cost=(sum(test_costs)/len(test_costs))
        print('Test:%d, Cost:%0.5f' % (pass_id, test_cost))
        if test_cost<min_cost:
            min_cost=test_cost
            fluid.io.save_inference_model(save_path, feeded_var_names=[image.name],target_vars=[result],executor=exe)
            patience=0
        else:
            patience=patience+1
            if patience==3:
                print("training is over!")
                break
