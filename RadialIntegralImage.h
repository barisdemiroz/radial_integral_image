#pragma once

#include <opencv2/opencv.hpp>
#include "geometry.h"

struct Pixel
{
	int x, y;
	double angle, radius;
	double val, sum;
};


class RadialIntegralImage
{
public:
	explicit RadialIntegralImage(const cv::Mat1f& img, const SectorPos& roi);
	double getSum(const AnnularSector& sector) const;
	const cv::Mat1f& getOriginalImg() const
	{
		return originalImg;
	}

	cv::Mat1f getIntegralImg() const
	{
		std::cout << "Warning expensive operation" << std::endl;
		return integralImg.clone();
	}

	static void testGetSum();

private:
	const cv::Mat1f originalImg;
	cv::Mat1d integralImg;
	cv::Mat1d twoPiRow;
	int radius;
	cv::Point topLeft;
	
	std::vector<Pixel> pixelsRaw;
	std::vector<Pixel*> pixels;
	
	void integral(int begin, int end, std::vector<Pixel*>& outPixels);
	void naiveIntegral(int begin, int end, std::vector<Pixel*>& outPixels);
	inline double lookup(double angle, double radius) const;
	
	inline double bilinear(double x, double y) const;

	inline double getSumNonCrossing(const AnnularSector& sector) const;
	double getSumNonCrossingRound(const AnnularSector& sector) const;
	double getSumNonCrossingInterp(const AnnularSector& sector) const;
	
	double getSumTwoPi(double angleMin, double radiusMin, double radiusMax) const;
	bool fullImage() const;
	void initTwoPiRow();
};

