# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:19:19 2017

@author: mjc13813759744
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%%
data_path = 'C:/D/anaconda/spyderlist/VGG/VGGweights/vgg19.npy'
data_dict = np.load(data_path, encoding='latin1').item()
#%%
# =============================================================================
# for key in data_dict.keys():
#     print('\n')
#     print(key)
#     print(data_dict[key][0].shape)
#     print(data_dict[key][1].shape)
# =============================================================================

#%%
def value(image,data_dict):
    value = {}
    x = tf.nn.conv2d(image,data_dict['conv1_1'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv1_1'][1])
    x = tf.nn.relu(x)
    value['conv1_1'] = x
    x = tf.nn.conv2d(x,data_dict['conv1_2'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv1_2'][1])
    x = tf.nn.relu(x)
    value['conv1_2'] = x
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    x = tf.nn.conv2d(x,data_dict['conv2_1'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv2_1'][1])
    x = tf.nn.relu(x)
    value['conv2_1'] = x
    x = tf.nn.conv2d(x,data_dict['conv2_2'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv2_2'][1])
    x = tf.nn.relu(x)
    value['conv2_2'] = x
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    x = tf.nn.conv2d(x,data_dict['conv3_1'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv3_1'][1])
    x = tf.nn.relu(x)
    value['conv3_1'] = x
    x = tf.nn.conv2d(x,data_dict['conv3_2'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv3_2'][1])
    x = tf.nn.relu(x)
    value['conv3_2'] = x
    x = tf.nn.conv2d(x,data_dict['conv3_3'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv3_3'][1])
    x = tf.nn.relu(x)
    value['conv3_3'] = x
    x = tf.nn.conv2d(x,data_dict['conv3_4'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv3_4'][1])
    x = tf.nn.relu(x)
    value['conv3_4'] = x
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    x = tf.nn.conv2d(x,data_dict['conv4_1'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv4_1'][1])
    x = tf.nn.relu(x)
    value['conv4_1'] = x
    x = tf.nn.conv2d(x,data_dict['conv4_2'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv4_2'][1])
    x = tf.nn.relu(x)
    value['conv4_2'] = x
    x = tf.nn.conv2d(x,data_dict['conv4_3'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv4_3'][1])
    x = tf.nn.relu(x)
    value['conv4_3'] = x
    x = tf.nn.conv2d(x,data_dict['conv4_4'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv4_4'][1])
    x = tf.nn.relu(x)
    value['conv4_4'] = x
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    x = tf.nn.conv2d(x,data_dict['conv5_1'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv5_1'][1])
    x = tf.nn.relu(x)
    value['conv5_1'] = x
    x = tf.nn.conv2d(x,data_dict['conv5_2'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv5_2'][1])
    x = tf.nn.relu(x)
    value['conv5_2'] = x
    x = tf.nn.conv2d(x,data_dict['conv5_3'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv5_3'][1])
    x = tf.nn.relu(x)
    value['conv5_3'] = x
    x = tf.nn.conv2d(x,data_dict['conv5_4'][0],strides=[1,1,1,1],padding='SAME')
    x = tf.nn.bias_add(x,data_dict['conv5_4'][1])
    x = tf.nn.relu(x)
    value['conv5_4'] = x
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    return value
#%%
def show_feature():
    image = np.random.rand(1,100,100,3)
    featuremap = prevalue(image,data_dict)
    feature_map = tf.reshape(featuremap['conv5_4'], [7,7,512])
    images = tf.image.convert_image_dtype (feature_map, dtype=tf.uint8)
    with tf.Session() as sess:
        images = sess.run(images)
        for i in np.arange(0, 5):
            plt.imshow(images[:,:,i])
            plt.show()
#%% 
def prevalue(image,data_dict):
    image = tf.cast(image,tf.float32)
    image = tf.reshape(image,[1,600,800,3])
    #with tf.Session() as sess:
        #image = sess.run(image)
    valcontent = value(image,data_dict)
    return valcontent

#%%
def contentcost(content,image):
    return (tf.reduce_sum(tf.square(tf.subtract(content['conv4_2'],image['conv4_2']))) / 2.0)
#%%
def stylecost(style,image):
    sess = tf.Session()
    n = style['conv1_1'].shape[3]
    m = style['conv1_1'].shape[1]*style['conv1_1'].shape[2]
    a = tf.reshape(style['conv1_1'],[m,n])
    a = tf.matmul(tf.transpose(a), a)
    b = tf.reshape(image['conv1_1'],[m,n])
    b = tf.matmul(tf.transpose(b), b)
    loss1 = (1. / (4 * n ** 2 * m ** 2)) * tf.reduce_sum(tf.square(a-b))   
    sess.close()
    n = style['conv2_1'].shape[3]
    m = style['conv2_1'].shape[1]*style['conv2_1'].shape[2]
    a = tf.reshape(style['conv2_1'],[m,n])
    a = tf.matmul(tf.transpose(a), a)
    b = tf.reshape(image['conv2_1'],[m,n])
    b = tf.matmul(tf.transpose(b), b)
    loss2 = (1. / (4 * n ** 2 * m ** 2)) * tf.reduce_sum(tf.square(a-b)) 
    
    n = style['conv3_1'].shape[3]
    m = style['conv3_1'].shape[1]*style['conv3_1'].shape[2]
    a = tf.reshape(style['conv3_1'],[m,n])
    a = tf.matmul(tf.transpose(a), a)
    b = tf.reshape(image['conv3_1'],[m,n])
    b = tf.matmul(tf.transpose(b), b)
    loss3 = (1. / (4 * n ** 2 * m ** 2)) * tf.reduce_sum(tf.square(a-b)) 

    n = style['conv4_1'].shape[3]
    m = style['conv4_1'].shape[1]*style['conv4_1'].shape[2]
    a = tf.reshape(style['conv4_1'],[m,n])
    a = tf.matmul(tf.transpose(a), a)
    b = tf.reshape(image['conv4_1'],[m,n])
    b = tf.matmul(tf.transpose(b), b)
    loss4 = (1. / (4 * n ** 2 * m ** 2)) * tf.reduce_sum(tf.square(a-b))  

    n = style['conv5_1'].shape[3]
    m = style['conv5_1'].shape[1]*style['conv5_1'].shape[2]
    a = tf.reshape(style['conv5_1'],[m,n])
    a = tf.matmul(tf.transpose(a), a)
    b = tf.reshape(image['conv5_1'],[m,n])
    b = tf.matmul(tf.transpose(b), b)
    loss5 = (1. / (4 * n ** 2 * m ** 2)) * tf.reduce_sum(tf.square(a-b))   
    return loss1 + loss2 + loss3 + loss4 + loss5
#%% 
def generate_image(content_image):
    content_image = np.reshape(content_image,[1,600,800,3])
    noise_image = np.random.uniform(0, 1,(1, 600, 800, 3)).astype('float32')
    img = noise_image * 0.6 + content_image * (1 - 0.6)
    return img
#%%
def train():
    sess = tf.Session()
    content_path = 'C:\\D\\anaconda\\spyderlist\\style_transfer\\content.jpg'
    content_image = plt.imread(content_path)
    #plt.imshow(content_image)
    content_value = prevalue(content_image,data_dict)   
    content_value = sess.run(content_value)
    
    style_path = 'C:\\D\\anaconda\\spyderlist\\style_transfer\\style.jpg'
    style_image = plt.imread(style_path)
    #plt.imshow(style_image)
    style_value = prevalue(style_image,data_dict) 
    style_value = sess.run(style_value)
    
    ximage = generate_image(content_image)
    ximage = tf.Variable(ximage,dtype = tf.float32)
    ximage_value = prevalue(ximage,data_dict)
    
    content_loss = contentcost(content_value,ximage_value)
    style_loss = stylecost(style_value,ximage_value)
    
    loss = content_loss + 100*style_loss
    
    optimizer = tf.train.AdamOptimizer(2.0).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    sess.run(init)
    allloss = []
    for i in range(100):
        pre_cost,_ = sess.run([loss,optimizer])
        print('interation:',i,'loss:',pre_cost)
        if (i+1)%10 == 0:
            allloss.append(pre_cost)
    sess.close()   
    return ximage,allloss
#%%
# =============================================================================
# content_path = 'C:\\D\\anaconda\\spyderlist\\style_transfer\\content.jpg'
# content_image = plt.imread(content_path)
# #plt.imshow(content_image)
# content_value = prevalue(content_image,data_dict)
# sess = tf.Session()
# a = sess.run(content_value)
# =============================================================================
#%%
image,loss = train() 
#%%                
x = image
#%%
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    xx = tf.reshape(x,[600,800,3])
    xxx = sess.run(xx)
    #plt.imshow(xxx)
    content_path = 'C:\\D\\anaconda\\spyderlist\\style_transfer\\content.jpg'
    content_image = plt.imread(content_path)   
    ximage = generate_image(content_image)
    print(ximage)
    ximage = np.reshape(ximage,[600,800,3])
    plt.imshow(ximage)    
        
        
        
        
        
        
        
        
        
        
        
        
        