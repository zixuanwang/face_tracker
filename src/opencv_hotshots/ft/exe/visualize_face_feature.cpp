#include "face_feature.h"
#include <iostream>
#include "opencv_hotshots/ft/ft.hpp"
#include "utilities.h"

// indices of landmarks
//int left_eye_indices[] = {32,72,33,73,34,74,35,75};
//int right_eye_indices[] = {27,71,30,70,29,69,28,68};
//int mouth_indices[] = {48,59,58,57,56,55,54,53,52,51,50,49};
// outer right eye, inner right eye, outer left eye, inner left eye
// right mouth, left mouth, nose
//int landmark_indices[] = {27,29,32,34,48,54,67};

cv::Mat camera_matrix, dist_coeffs;
cv::Mat rvec, tvec;

//void head_pose(const std::vector<cv::Point2f>& points){
//	std::vector<cv::Point2f> image_array;
//	std::vector<cv::Point3f> model_array;
//	int landmark_indices_size = sizeof(landmark_indices) / sizeof(int);
//	for(int i = 0; i < landmark_indices_size; ++i){
//		image_array.push_back(points[landmark_indices[i]]);
//		model_array.push_back(cv::Point3f(model_points[3 * i], model_points[3 * i + 1], model_points[3 * i + 2]));
//	}
//	cv::solvePnP(model_array, image_array, camera_matrix, dist_coeffs, rvec, tvec);
//	//std::cout << rvec << std::endl;
//	std::cout << tvec << std::endl;
//}

void load_camera_model(const std::string& filename){
	cv::FileStorage f(filename, cv::FileStorage::READ);
	f["cameraMatrix"] >> camera_matrix;
	f["distCoeffs"] >> dist_coeffs;
	f.release();
}

#define fl at<float>
const char* usage = "usage: ./visualise_face_feature camera_model tracker_model [video_file]";
//==============================================================================
void
draw_string(Mat img,                       //image to draw on
        const string text)             //text to draw
{
  Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
  putText(img,text,Point(0,size.height),FONT_HERSHEY_COMPLEX,0.6f,
      CV_RGB(200,50,50),1,CV_AA);
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
	face_feature face;

  //parse command line arguments
  if(parse_help(argc,argv)){cout << usage << endl; return 0;}
  if(argc < 3){cout << usage << endl; return 0;}
  	//cv::VideoWriter video_writer;
	//video_writer.open(argv[4], CV_FOURCC('M','P','4','2'), 30, cv::Size(640, 480));

  //load detector model
  face_tracker tracker = load_ft<face_tracker>(argv[2]);

  //create tracker parameters
  face_tracker_params p; p.robust = false;
  p.ssize.resize(3);
  p.ssize[0] = Size(21,21);
  p.ssize[1] = Size(11,11);
  p.ssize[2] = Size(5,5);

  //open video stream
  VideoCapture cam; 
  if(argc > 2)cam.open(atoi(argv[3])); else cam.open(0);
  if(!cam.isOpened()){
    cout << "Failed opening video file." << endl
     << usage << endl; return 0;
  }
  //detect until user quits
  namedWindow("face feature");
  //load camera model
  load_camera_model(argv[1]);
  while(cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){
    Mat im; cam >> im; 
	//video_writer << im; // save frame.
    if(tracker.track(im,p)){
		face.calc_count(tracker.points);
		//head_pose(tracker.points);
		tracker.draw(im);
	}
	draw_string(im,"blink: " + boost::lexical_cast<std::string>(face.get_blink_count()) + "  yawn: " + boost::lexical_cast<std::string>(face.get_yawn_count()));
    tracker.timer.display_fps(im,Point(1,im.rows-1));
    imshow("face feature",im);
    int c = waitKey(10);
    if(c == 'q')break;
	else if(c == 'd'){
		face.set_blink_count(0);
		face.set_yawn_count(0);
		tracker.reset();
	}
  }
  destroyWindow("face feature"); cam.release(); return 0;
	return 0;
}