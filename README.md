# Joint-Bilateral-Filter
 Implement Joint bilateral filter to identify the best weight to convert a color image to a grayscale image.

1. Given a color image, a set of weight combination corresponds to a grayscale image candidate.
2. Measuring the perceptual similarity by Joint bilateral filter.

Original Image        |  Grayscale image with highest cost        |  Grayscale image with lowest cost
:-------------------------:|:-------------------------:|:------------------:
![](https://github.com/ronnie0726/Joint-Bilateral-Filter/blob/main/testdata/1.png)  |    ![](https://github.com/ronnie0726/Joint-Bilateral-Filter/blob/main/result/max_error_gray.png) |   ![](https://github.com/ronnie0726/Joint-Bilateral-Filter/blob/main/result/low_error_gray.png)
