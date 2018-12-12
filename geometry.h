#pragma once

#include <opencv2/opencv.hpp>

class AnnularSector
{
public:
	AnnularSector(double angleMin, double angleMax, double radiusMin, double radiusMax)
		: angleMin(angleMin), angleMax(angleMax), radiusMin(radiusMin), radiusMax(radiusMax)
	{
		assert(angleMin < angleMax);
	}

	double dAngle() const
	{
		return angleMax - angleMin;
	}

	double dRadius() const
	{
		return radiusMax - radiusMin;
	}

	bool contains(int x, int y) const
	{
		double r, angle;
		cartToPolar(x, y, r, angle);
		if (angleMax > 2 * M_PI)
			return radiusMin <= r && r < radiusMax && ((angleMin <= angle && angle < 2 * M_PI) || (0 <= angle && angle < angleMax - 2 * M_PI));
		return radiusMin <= r && r < radiusMax && angleMin <= angle && angle < angleMax;
	}

	double angleMin, angleMax;
	double radiusMin, radiusMax;
};


class SectorPos
{
public:
	SectorPos(int radius, const cv::Point& topLeft)
		: radius(radius), topLeft(topLeft))
	{
	}

	int radius;
	cv::Point topLeft;
};

