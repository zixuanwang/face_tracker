#include "utilities.h"

long long timer(){
	auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

float polygon_area(const std::vector<cv::Point2f>& points, int* indices, int n){
	boost::geometry::model::polygon<boost::tuple<float, float> > polygon;
	for(int i = 0; i < n; ++i){
		boost::geometry::append(polygon,boost::make_tuple(points[indices[i]].x,points[indices[i]].y));
	}
	return boost::geometry::area(polygon);
}