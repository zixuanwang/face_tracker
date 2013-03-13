#pragma once
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/lexical_cast.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)
long long timer();
float polygon_area(const std::vector<cv::Point2f>& points, int* indices, int n);