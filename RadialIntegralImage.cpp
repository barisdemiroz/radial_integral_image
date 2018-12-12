#include "RadialIntegralImage.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define M_2PI (2*M_PI)

using namespace cv;

Mat1f extraSlowComputeIntegralImage(const vector<Pixel*>& pixels, const Size& size)
{
	size_t nPoints = pixels.size();
	Mat1d integralImg = Mat1d::zeros(size);

	for (int i = 0; i < nPoints; ++i)
	{
		double sum = 0;
		for (int j = 0; j < nPoints; ++j)
		{
			if (pixels[i]->radius >= pixels[j]->radius && pixels[i]->angle >= pixels[j]->angle)
				sum += pixels[j]->val;
		}
		integralImg(pixels[i]->y, pixels[i]->x) = sum;
	}

	return integralImg;
}


Mat1d slowComputeIntegralImage(const vector<Pixel*>& pixels, const Size& size)
{
	size_t nPoints = pixels.size();
	Mat1d integralImg = Mat1d::zeros(size);

	for (int i = nPoints - 1; i >= 0; --i)
	{
		double sum = 0;
		for (int j = i; j >= 0; --j)
		{
			if (pixels[i]->radius >= pixels[j]->radius && pixels[i]->angle >= pixels[j]->angle)
				sum += pixels[j]->val;
		}
		integralImg(pixels[i]->y, pixels[i]->x) = sum;
	}

	return integralImg;
}

RadialIntegralImage::RadialIntegralImage(const Mat1f& img, const SectorPos& roi)
	: originalImg{img}, radius{ roi.radius }, topLeft{ roi.topLeft }
{
	int nPoints = img.rows * img.cols;

	pixelsRaw.resize(nPoints);

	int n = 0;
	for (int y = 0; y < img.rows; ++y)
	{
		for (int x = 0; x < img.cols; ++x)
		{
			int i = y + roi.topLeft.y - roi.radius;
			int j = x + roi.topLeft.x - roi.radius;

			Pixel& p = pixelsRaw[n++];
			p.x = x;
			p.y = y;
			p.angle = fastAtan2(i, j) * M_PI / 180;

			p.radius = sqrt(i*i + j*j);
			p.val = img(y, x);
		}
	}

	pixels.resize(nPoints);
	for (int i = 0; i < nPoints; ++i)
		pixels[i] = &pixelsRaw[i];

	sort(pixels.begin(), pixels.end(), [](const Pixel* a, const Pixel* b) -> bool {
		if (a->radius == b->radius)
			return a->angle < b->angle;
		return a->radius < b->radius;
	});

	integralImg = Mat1d::zeros(img.size());

	vector<Pixel*> tmp(nPoints);
	integral(0, nPoints, tmp);
	
	initTwoPiRow();

	//	{ // DEBUG
	//		Mat1f slowImg = slowComputeIntegralImage(pixels, integralImg.size());
	//		Mat1f extraSlowImg = extraSlowComputeIntegralImage(pixels, integralImg.size());
	//
	//		Mat1f saglamaDiff = abs(slowImg - extraSlowImg) / extraSlowImg;
	//		Mat1f slowDiff = abs(slowImg - integralImg) / slowImg;
	//		Mat1f extraSlowDiff = abs(extraSlowImg - integralImg) / extraSlowImg;
	//		double saglamaMax, slowMax, extraSlowMax;
	//		minMaxLoc(saglamaDiff, nullptr, &saglamaMax);
	//		minMaxLoc(slowDiff, nullptr, &slowMax);
	//		minMaxLoc(extraSlowDiff, nullptr, &extraSlowMax);
	//		std::cout << "\n";
	//	}
}


void RadialIntegralImage::initTwoPiRow()
{
	if (!fullImage())
		return;

	twoPiRow.create(1, radius - 1);

	Mat foo = integralImg.row(radius).colRange(radius + 1, integralImg.cols);
	foo.copyTo(twoPiRow);


	double acc = 0;

	int i = 0, j = 0;
	while (i < radius - 1 && j < pixels.size())
	{
		int r = i + 1;
		if (pixels[j]->radius <= r)
		{
			acc += pixels[j]->val;
			++j;
		}
		else
		{
			twoPiRow(i) += acc;
			++i;
		}
	}

	for (; i < radius - 1; ++i)
		twoPiRow(i) += acc;
}


void RadialIntegralImage::naiveIntegral(int begin, int end, std::vector<Pixel*>& outPixels)
{
	std::partial_sort_copy(pixels.begin() + begin, pixels.begin() + end, outPixels.begin(), outPixels.end(), [](const Pixel* a, const Pixel* b) -> bool {
		if (a->angle == b->angle)
			return a->radius < b->radius;
		return a->angle < b->angle;
	});
	
	int n = end-begin;
	for (int i = n-1; i >= 0; --i)
	{
		double acc = outPixels[i]->val;
		for (int j = i-1; j >= 0; --j)
		{
			if (outPixels[i]->angle >= outPixels[j]->angle && outPixels[i]->radius >= outPixels[j]->radius)
				acc += outPixels[j]->val;
		}
		outPixels[i]->sum = acc;
		integralImg(outPixels[i]->y, outPixels[i]->x) = acc;
	}
}


void RadialIntegralImage::integral(int begin, int end, vector<Pixel*>& outPixels)
{
	if (end - begin <= 16)
	{
		naiveIntegral(begin, end, outPixels);
		return;
	}

	if (begin + 1 == end)
	{
		pixels[begin]->sum = pixels[begin]->val;
		integralImg(pixels[begin]->y, pixels[begin]->x) = pixels[begin]->val;
		outPixels[0] = pixels[begin];
		return;
	}

	int pivot = (begin + end) / 2;

	vector<Pixel*> pixelsA(pivot-begin), pixelsB(end-pivot);

	integral(begin, pivot, pixelsA);
	integral(pivot, end, pixelsB);

	double acc = 0;

	int i = 0, j = 0, k = 0;
	for (; i < pixelsA.size() && j < pixelsB.size(); ++k)
	{
		if (pixelsA[i]->angle <= pixelsB[j]->angle && pixelsA[i]->radius <= pixelsB[j]->radius)
		{
			acc += pixelsA[i]->val;
			outPixels[k] = pixelsA[i];
			++i;
		}
		else
		{
			pixelsB[j]->sum += acc;
			integralImg(pixelsB[j]->y, pixelsB[j]->x) = pixelsB[j]->sum;
			outPixels[k] = pixelsB[j];
			++j;
		}
	}

	for (; i < pixelsA.size(); ++i, ++k)
		outPixels[k] = pixelsA[i];

	for (; j < pixelsB.size(); ++j, ++k)
	{
		pixelsB[j]->sum += acc;
		integralImg(pixelsB[j]->y, pixelsB[j]->x) = pixelsB[j]->sum;
		outPixels[k] = pixelsB[j];
	}
}


double RadialIntegralImage::getSum(const AnnularSector& sector) const
{
	assert(sector.angleMax < 2 * M_2PI);

	if (sector.angleMax <= M_2PI)
		return getSumNonCrossing(sector);

	if (sector.angleMin > M_2PI)
		return getSumNonCrossing({sector.angleMin - M_2PI, sector.angleMax - M_2PI, sector.radiusMin, sector.radiusMax});

	AnnularSector sector2(0, sector.angleMax - M_2PI, sector.radiusMin, sector.radiusMax);
	return getSumTwoPi(sector.angleMin, sector.radiusMin, sector.radiusMax) + getSumNonCrossing(sector2);
}

inline double bilinear_unit(double x, double y, double f00, double f10, double f01, double f11)
{
	double a00 = f00;
	double a10 = f10 - f00;
	double a01 = f01 - f00;
	double a11 = f11 + f00 - f10 - f01;
	return a00 + a10 * x + a01 * y + a11 * x * y;
}

double RadialIntegralImage::bilinear(double x, double y) const
{
	int x0 = x, y0 = y;
	int x1 = x0 + 1, y1 = y0 + 1;
	return bilinear_unit(x - x0, y - y0, integralImg(y0, x0), integralImg(y0, x1), integralImg(y1, x0), integralImg(y1, x1));
}


inline int fast_round(double r) {
	return (r > 0.0) ? (r + 0.5) : (r - 0.5);
}

double RadialIntegralImage::lookup(double angle, double radius) const
{
	assert(angle > 0 && angle < M_2PI && radius > 0);

	int x = fast_round(radius * cos(angle)) - topLeft.x + this->radius;
	int y = fast_round(radius * sin(angle)) - topLeft.y + this->radius;

	return integralImg(y, x);
}

double RadialIntegralImage::getSumTwoPi(double angleMin, double radiusMin, double radiusMax) const
{
	radiusMax = min(radiusMax, double(radius - 1));

	int xa = fast_round(radiusMin * cos(angleMin)) - topLeft.x + radius;
	int ya = fast_round(radiusMin * sin(angleMin)) - topLeft.y + radius;
	int xb = fast_round(radiusMax * cos(angleMin)) - topLeft.x + radius;
	int yb = fast_round(radiusMax * sin(angleMin)) - topLeft.y + radius;

	double a = (ya == radius) ? twoPiRow(fast_round(radiusMin) - 1) : integralImg(ya, xa);
	double b = (yb == radius) ? twoPiRow(fast_round(radiusMax) - 1) : integralImg(yb, xb);
	double c = twoPiRow(fast_round(radiusMax)-1);
	double d = twoPiRow(fast_round(radiusMin)-1);

	return c - b - d + a;
}

bool RadialIntegralImage::fullImage() const
{
	return integralImg.cols/2 == radius && integralImg.rows/2 == radius;
}

double RadialIntegralImage::getSumNonCrossing(const AnnularSector& sector) const
{
	return getSumNonCrossingInterp(sector);
//	return getSumNonCrossingRound(sector);
}

double RadialIntegralImage::getSumNonCrossingInterp(const AnnularSector& sector) const
{
	double rmax = min(sector.radiusMax, double(radius - 1)), rmin = sector.radiusMin;
	double cmin = cos(sector.angleMin), smin = sin(sector.angleMin);
	double cmax = cos(sector.angleMax), smax = sin(sector.angleMax);
	double xa = rmin * cmin - topLeft.x + radius;
	double ya = rmin * smin - topLeft.y + radius;
	double xb = rmax * cmin - topLeft.x + radius;
	double yb = rmax * smin - topLeft.y + radius;
	double xc = rmax * cmax - topLeft.x + radius;
	double yc = rmax * smax - topLeft.y + radius;
	double xd = rmin * cmax - topLeft.x + radius;
	double yd = rmin * smax - topLeft.y + radius;

	double a = bilinear(xa, ya);
	double b = bilinear(xb, yb);

	double c = (yc + topLeft.y == radius && sector.angleMax > 0.75 * M_2PI) ? twoPiRow(fast_round(rmax) - 1) : bilinear(xc, yc);
	double d = (yd + topLeft.y == radius && sector.angleMax > 0.75 * M_2PI) ? twoPiRow(fast_round(rmin) - 1) : bilinear(xd, yd);

	return c - b - d + a;
}


double RadialIntegralImage::getSumNonCrossingRound(const AnnularSector& sector) const
{
	double rmax = min(sector.radiusMax, double(radius-1)), rmin = sector.radiusMin;
	double cmin = cos(sector.angleMin), smin = sin(sector.angleMin);
	double cmax = cos(sector.angleMax), smax = sin(sector.angleMax);
	int xa = fast_round(rmin * cmin) - topLeft.x + radius;
	int ya = fast_round(rmin * smin) - topLeft.y + radius;
	int xb = fast_round(rmax * cmin) - topLeft.x + radius;
	int yb = fast_round(rmax * smin) - topLeft.y + radius;
	int xc = fast_round(rmax * cmax) - topLeft.x + radius;
	int yc = fast_round(rmax * smax) - topLeft.y + radius;
	int xd = fast_round(rmin * cmax) - topLeft.x + radius;
	int yd = fast_round(rmin * smax) - topLeft.y + radius;
	
//	{
//		Mat3f img = Mat3f::zeros(512,512);
//		Vec3f val(0.3,0.8,0.9);
//		img(ya + topLeft.y, xa + topLeft.x) = val;
//		img(yb + topLeft.y, xb + topLeft.x) = val;
//		img(yc + topLeft.y, xc + topLeft.x) = val;
//		img(yd + topLeft.y, xd + topLeft.x) = val;
//		rectangle(img, Rect(topLeft, integralImg.size()), Scalar(0.9, 0.3, 0.8));
//		waitKey(1);
//	}

//	{
//		Mat img;
//		originalImg.convertTo(img, CV_32F);
//		cvtColor(img, img, CV_GRAY2BGR);
//		Vec3f val(0.3*255, 0.8*255, 0.9*255);
//		img.at<Vec3f>(ya, xa) = val;
//		img.at<Vec3f>(yb, xb) = val;
//		img.at<Vec3f>(yc, xc) = val;
//		img.at<Vec3f>(yd, xd) = val;
//		waitKey(1);
//	}

	double a = integralImg(ya, xa);
	double b = integralImg(yb, xb);

	double c = (yc + topLeft.y == radius && sector.angleMax > 0.75 * M_2PI) ? twoPiRow(fast_round(rmax)-1) : integralImg(yc, xc);
	double d = (yd + topLeft.y == radius && sector.angleMax > 0.75 * M_2PI) ? twoPiRow(fast_round(rmin)-1) : integralImg(yd, xd);
//	double c = integralImg(yc, xc);
//	double d = integralImg(yd, xd);

	return c - b - d + a;
}


void RadialIntegralImage::testGetSum()
{
	vector<double> errors;

	Mat1d randMat(512, 512);
	for (int i = 0; i < 100; ++i)
	{
		theRNG().fill(randMat, RNG::UNIFORM, 1.0, 100.0);

		RadialIntegralImage integralImg(randMat, SectorPos{ 256, { 0, 0 }, 72 });

		double width = theRNG().uniform(0.1, 1.0);
		double angleMin = theRNG().uniform(0.0, M_2PI-width);
		
		int height = theRNG().uniform(1, 100);
		int rMin = theRNG().uniform(5, 250-height);
		AnnularSector sector{ angleMin, angleMin + width, double(rMin), double(rMin + height) };

		double actual = integralImg.getSum(sector);

		double expected = 0;
		for (Pixel& pixel : integralImg.pixelsRaw)
		{
			if (sector.angleMin < pixel.angle && pixel.angle <= sector.angleMax &&
				sector.radiusMin < pixel.radius && pixel.radius <= sector.radiusMax)
				expected += pixel.val;
		}

		double diff = expected - actual;
		double err = diff / expected;
		printf("%d\n", i);
		printf("a:[%.2f - %.2f]  r:[%.2f - %.2f]   da:%.2f   dr:%.2f\n", sector.angleMin, sector.angleMax, sector.radiusMin, sector.radiusMax, sector.dAngle(), sector.dRadius());
		printf("expected:%.2f   actual:%.2f   diff:%.2f  rel_err:%.2f\n\n", expected, actual, diff, err);

		errors.push_back(err);
	}

	for (double e : errors)
		std::cout << e << ",";
}

