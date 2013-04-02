#include "face_feature.h"


face_feature::face_feature(void)
{
	// hard coding indices.
	int _left_eye_indices[] = {4,5,6,7};
	int _right_eye_indices[] = {0,1,2,3};
	int _mouth_indices[] = {14,15,16,17,18,19,20,21};
	int _nose_indices[] = {8,9,10,11,12};
	// outer right eye, inner right eye, outer left eye, inner left eye
	// right mouth, left mouth, nose
	int _landmark_indices[] = {0,2,6,4,14,18,13};
	float _model_points[] = {0.045f,0.035f,0.0f,0.015f,0.035f,0.005f,-0.045f,0.035f,0.0f,-0.015f,0.035f,0.005f,0.03f,-0.03f,0.0f,-0.03f,-0.03f,0.0f,0.0f,0.0f,0.035f};
	left_eye_indices.assign(_left_eye_indices, _left_eye_indices + sizeof(_left_eye_indices) / sizeof(_left_eye_indices[0]));
	right_eye_indices.assign(_right_eye_indices, _right_eye_indices + sizeof(_right_eye_indices) / sizeof(_right_eye_indices[0]));
	mouth_indices.assign(_mouth_indices, _mouth_indices + sizeof(_mouth_indices) / sizeof(_mouth_indices[0]));
	nose_indices.assign(_nose_indices, _nose_indices + sizeof(_nose_indices) / sizeof(_nose_indices[0]));
	landmark_indices.assign(_landmark_indices, _landmark_indices + sizeof(_landmark_indices) / sizeof(_landmark_indices[0]));
	model_points.assign(_model_points, _model_points + sizeof(_model_points) / sizeof(_model_points[0]));
	m_blink_count = 0;
	m_yawn_count = 0;
	m_blink_trigger = false;
	m_yawn_trigger = false;
}


face_feature::~face_feature(void)
{
}

void face_feature::calc_count(const std::vector<cv::Point2f>& points){
	float nose_area = polygon_area(points, nose_indices); // used as reference area.
	float mouth_area = polygon_area(points, mouth_indices);
	float eye_area = polygon_area(points, left_eye_indices) + polygon_area(points, right_eye_indices);
	m_mouth_area_list.push_back(mouth_area / nose_area);
	m_eye_area_list.push_back(eye_area / nose_area);
	int size = 5;
	int middle = size / 2;
	if(static_cast<int>(m_mouth_area_list.size()) > size && static_cast<int>(m_eye_area_list.size()) > size){
		m_mouth_area_list.pop_front();
		m_eye_area_list.pop_front();
		std::vector<float> mouth_area_vector(m_mouth_area_list.begin(), m_mouth_area_list.end());
		std::vector<float> eye_area_vector(m_eye_area_list.begin(), m_eye_area_list.end());
		std::nth_element(mouth_area_vector.begin(), mouth_area_vector.begin() + middle, mouth_area_vector.end());
		float median_mouth_area = mouth_area_vector[middle];
		std::nth_element(eye_area_vector.begin(), eye_area_vector.begin() + middle, eye_area_vector.end());
		float median_eye_area = eye_area_vector[middle];
		// thresholding
		if(median_mouth_area > 1.6f){
			m_yawn_trigger = true;
		}else if(m_yawn_trigger){
			m_yawn_trigger = false;
			++m_yawn_count;
		}
		if(median_eye_area < 0.9f){
			m_blink_trigger = true;
		}else if(m_blink_trigger){
			m_blink_trigger = false;
			++m_blink_count;
		}
	}
	m_distance = 4215.2f / sqrtf(nose_area);
}

float face_feature::polygon_area(const std::vector<cv::Point2f>& points, const std::vector<int>& indices){
	boost::geometry::model::polygon<boost::tuple<float, float> > polygon;
	for(size_t i = 0; i < indices.size(); ++i){
		boost::geometry::append(polygon,boost::make_tuple(points[indices[i]].x,points[indices[i]].y));
	}
	return boost::geometry::area(polygon);
}

float face_feature::get_distance(){
	return m_distance;
}
