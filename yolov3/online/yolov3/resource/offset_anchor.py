#coding:utf-8
import sys
import os
import math

# 和训练cfg文件一致
input_size = 608
anchor=[10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

output_size = int(sys.argv[1])
# 76 38 19 -> 8 16 32
idx = int(math.log(input_size//output_size, 2) - 3)*3
ratio = output_size*1. / input_size

anchor1=[anchor[idx]*ratio,anchor[idx+1]*ratio ]
anchor2=[anchor[idx+2]*ratio,anchor[idx+3]*ratio ]
anchor3=[anchor[idx+4]*ratio,anchor[idx+5]*ratio ]

anchors_tensor_fname = os.path.join("./","anchors_tensor_{}".format(output_size))
x_y_offset_fname = os.path.join("./","x_y_offset_file_{}".format(output_size))
with open(anchors_tensor_fname, "w") as anchors_tensor_f, open(x_y_offset_fname, "w") as x_y_offset_f:
    for row in range(output_size):
        for colum in range(output_size):
            x_y_offset_f.write(str(colum)+"\t"+str(row)+'\n'+str(colum)+"\t"+str(row)+'\n'+str(colum)+"\t"+str(row)+'\n')

            anchors_tensor_f.write(str(anchor1[0])+"\t"+str(anchor1[1])+'\n'+str(anchor2[0])+"\t"+str(anchor2[1])+'\n'+str(anchor3[0])+"\t"+str(anchor3[1])+'\n')