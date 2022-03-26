import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ### TODO ###
    gray_img = []
    gray_img.append(img_gray)
    f = open(args.setting_path, 'r')
    l = []
    for line in f.readlines():
        l.append(line.split(','))
    f.close()
    for line in l:
        for i in range(len(line)):
            line[i] = line[i].replace('\n','')
    
    w = l[1:len(l)-1]
    sigma_s = int(l[-1][1])
    sigma_r = float(l[-1][3])
    for i in range(5):
        gray_img.append(float(w[i][0])*img_rgb[:,:,0] + float(w[i][1])*img_rgb[:,:,1] + float(w[i][2])*img_rgb[:,:,2])
    
    # create JBF class
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    error = []
    filtered_rgb = []
    for guidance in gray_img:
        jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
        e = np.sum(np.abs(jbf_out.astype('int32')-bf_out.astype('int32')))
        jbf_out = cv2.cvtColor(jbf_out,cv2.COLOR_RGB2BGR)
        filtered_rgb.append(jbf_out)
        error.append(e)
    for i in range(len(error)):
         if error[i] == max(error):
             cv2.imwrite('./result/max_error_gray.png',gray_img[i])
             cv2.imwrite('./result/max_error_rgb.png',filtered_rgb[i])

         elif error[i] == min(error):
             cv2.imwrite('./result/low_error_gray.png',gray_img[i])
             cv2.imwrite('./result/low_error_rgb.png',filtered_rgb[i])


if __name__ == '__main__':
    main()
