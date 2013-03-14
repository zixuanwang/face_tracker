#pragma once
#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)
class face_feature
{
public:
	face_feature(void);
	~face_feature(void);
	void calc_count(const std::vector<cv::Point2f>& points);
	float polygon_area(const std::vector<cv::Point2f>& points, const std::vector<int>& indices);
	int get_blink_count(){return m_blink_count;}
	int get_yawn_count(){return m_yawn_count;}
	void set_blink_count(int blink_count){m_blink_count = blink_count;}
	void set_yawn_count(int yawn_count){m_yawn_count = yawn_count;}
private:
	std::vector<int> left_eye_indices;
	std::vector<int> right_eye_indices;
	std::vector<int> mouth_indices;
	std::vector<int> nose_indices;
	std::vector<int> landmark_indices;
	std::vector<float> model_points;
	int m_blink_count;
	int m_yawn_count;
	bool m_blink_trigger;
	bool m_yawn_trigger;
	std::list<float> m_mouth_area_list;
	std::list<float> m_eye_area_list;
};

