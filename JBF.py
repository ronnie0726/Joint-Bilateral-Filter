import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        x,y = np.mgrid[-self.pad_w:self.pad_w+1, -self.pad_w:self.pad_w+1]
        s_kernel = np.exp((x**2+y**2)/(-2*(self.sigma_s**2)))

        padded_guidance = padded_guidance/255

        output = np.zeros(img.shape)

        for i in range(self.pad_w, padded_guidance.shape[0]-self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1]-self.pad_w):
                w = padded_guidance[i-self.pad_w:i+self.pad_w+1,j-self.pad_w:j+self.pad_w+1]
                r_kernel = ((w - padded_guidance[i,j])**2)/(-2*(self.sigma_r**2))
                if len(padded_guidance.shape) == 3:
                    r_kernel = np.sum(r_kernel,axis = 2)
                r_kernel = np.exp(r_kernel)
                
                g = s_kernel*r_kernel
                g /= np.sum(g)
                
                for c in range(3):
                    output[i-self.pad_w,j-self.pad_w,c] = np.sum(g*padded_img[i-self.pad_w:i+self.pad_w+1,j-self.pad_w:j+self.pad_w+1,c])
       
        return np.clip(output, 0, 255).astype(np.uint8)