#include "opencv_hotshots/ft/ft.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>

// indices of landmarks
int left_eye_indices[] = {32,72,33,73,34,74,35,75};
int right_eye_indices[] = {27,71,30,70,29,69,28,68};
int mouth_indices[] = {48,59,58,57,56,55,54,53,52,51,50,49};

// outer right eye, inner right eye, outer left eye, inner left eye
// right mouth, left mouth, nose
int landmark_indices[] = {27,29,32,34,48,54,67};
// test model_points
float model_points[] = {0.07,0.07,0.0,0.03,0.07,0.0,-0.07,0.07,0.0,-0.03,0.07,0.0,0.04,-0.04,0.005,-0.04,-0.04,0.005,0.0,0.0,0.03};

cv::Mat camera_matrix, dist_coeffs;

BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)

float polygon_area(const std::vector<cv::Point2f>& points, int* indices, int n){
	boost::geometry::model::polygon<boost::tuple<float, float> > polygon;
	for(int i = 0; i < n; ++i){
		boost::geometry::append(polygon,boost::make_tuple(points[indices[i]].x,points[indices[i]].y));
	}
	return boost::geometry::area(polygon);
}

void head_pose(const std::vector<cv::Point2f>& points, int* indices, int n){
	std::vector<cv::Point2f> image_array;
	std::vector<cv::Point3f> model_array;
	for(int i = 0; i < n; ++i){
		image_array.push_back(points[indices[i]]);
		model_array.push_back(cv::Point3f(model_points[3 * i], model_points[3 * i + 1], model_points[3 * i + 2]));
	}
	cv::Mat rvec, tvec;
	cv::solvePnP(model_array, image_array, camera_matrix, dist_coeffs, rvec, tvec);
	//std::cout << rvec << std::endl;
	//std::cout << tvec << std::endl;
}

void load_camera_model(const std::string& filename){
	cv::FileStorage f(filename, cv::FileStorage::READ);
	f["cameraMatrix"] >> camera_matrix;
	f["distCoeffs"] >> dist_coeffs;
	f.release();
}

#define fl at<float>
const char* usage = "usage: ./visualise_face_tracker tracker [video_file]";
//==============================================================================
void
draw_string(Mat img,                       //image to draw on
        const string text)             //text to draw
{
  Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
  putText(img,text,Point(0,size.height),FONT_HERSHEY_COMPLEX,0.6f,
      Scalar::all(0),1,CV_AA);
  putText(img,text,Point(1,size.height+1),FONT_HERSHEY_COMPLEX,0.6f,
      Scalar::all(255),1,CV_AA);
}
//==============================================================================
bool
parse_help(int argc,char** argv)
{
  for(int i = 1; i < argc; i++){
    string str = argv[i];
    if(str.length() == 2){if(strcmp(str.c_str(),"-h") == 0)return true;}
    if(str.length() == 6){if(strcmp(str.c_str(),"--help") == 0)return true;}
  }return false;
}
//==============================================================================
int main(int argc,char** argv)
{
  //parse command line arguments
  if(parse_help(argc,argv)){cout << usage << endl; return 0;}
  if(argc < 2){cout << usage << endl; return 0;}
  
  //load detector model
  face_tracker tracker = load_ft<face_tracker>(argv[1]);

  //create tracker parameters
  face_tracker_params p; p.robust = false;
  p.ssize.resize(3);
  p.ssize[0] = Size(21,21);
  p.ssize[1] = Size(11,11);
  p.ssize[2] = Size(5,5);

  //open video stream
  VideoCapture cam; 
  if(argc > 2)cam.open(atoi(argv[2])); else cam.open(0);
  if(!cam.isOpened()){
    cout << "Failed opening video file." << endl
     << usage << endl; return 0;
  }
  //detect until user quits
  namedWindow("face tracker");
  //load camera model
  load_camera_model("C:/Users/Zixuan/data/camera/microsoft_laptop.xml");
  while(cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){
    Mat im; cam >> im; 
    if(tracker.track(im,p)){
		std::cout << polygon_area(tracker.points, mouth_indices, sizeof(mouth_indices) / sizeof(int)) << std::endl;
		head_pose(tracker.points, landmark_indices, sizeof(landmark_indices) / sizeof(int));
		tracker.draw(im);
	}
    draw_string(im,"d - redetection");
    tracker.timer.display_fps(im,Point(1,im.rows-1));
    imshow("face tracker",im);
    int c = waitKey(10);
    if(c == 'q')break;
    else if(c == 'd')tracker.reset();
  }
  destroyWindow("face tracker"); cam.release(); return 0;
}