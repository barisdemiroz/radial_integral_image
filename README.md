# Radial integral image

Radial integral image allows rapidly calculating pixel sums inside annular sectors (doughnut slice shapes) for a given image.

<img alt="Radial integral image" src="https://github.com/barisdemiroz/radial_integral_image/raw/master/radial_integral_image.png" width="400" />


Why?
----
During my PhD studies I came up with a scheme that allows to form radial integral images and used that for person detection in omnidirectional images. Using [integral images](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework#Summed_area_table) is a popular way of speeding up computations in computer vision algorithms.


Usage
-----
First you construct the integral image by passing an image and a roi (region of interest). roi allows you to form a partial integral image just for the specified part of the image.
```cpp
Mat1f img;
// ... load the image

SectorPos(radius, Point(0, 0)); // initialize radial integral image for the whole image

RadialIntegralImage integralImg(img, pos);
```

Then you can query sums on the integral image:
```cpp
AnnularSector sector(M_PI/5, M_PI/3, 30, 50);
double sum = integralImg.getSum(sector);
```


Remarks
-------
I have extracted these files from a project, so it contains some cruft. There is a OpenCV 2.4+ dependency.

If you found this code useful please cite: "Demiröz, B. E., Salah, A. A., Bastanlar, Y., & Akarun, L. (2019). **Affordable person detection in omnidirectional cameras using radial integral channel features**. Machine Vision and Applications, 30(4), 645-655."
