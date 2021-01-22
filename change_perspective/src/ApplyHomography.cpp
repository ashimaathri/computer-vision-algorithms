#include <mvg.h>
#include <math.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

Mat construct_roi_mask(int height, int width, vector<Point2f> points) {
  Mat mask(height, width, CV_8UC1, cv::Scalar(0));
  vector<vector<Point>> contours;
  contours.push_back(vector<Point>());
  // Careful about the order of the points defining the ROI
  contours[0].push_back(points[0]);
  contours[0].push_back(points[1]);
  contours[0].push_back(points[3]);
  contours[0].push_back(points[2]);
  drawContours(mask, contours, 0, Scalar(255), FILLED, 8);
  return mask;
}

Mat copyPerspectively(Mat to, Mat from, Mat homography, std::vector <Point2f> points) {
  int dstHeight = to.size().height;
  int dstWidth = to.size().width;
  int srcHeight = from.size().height;
  int srcWidth = from.size().width;

  Mat dst_x = Mat(dstHeight, dstWidth, CV_32FC1);
  Mat dst_y = Mat(dstHeight, dstWidth, CV_32FC1);
  Mat transformed_src = Mat(dstHeight, dstWidth, to.type());

  for(int x = 0; x < dstWidth; x++) {
    for(int y = 0; y < dstHeight; y++) {
      Point2f src_point = transform_point(homography, x, y);
      // Effectively doesn't interpolate the borders and replaces them with 0s
      if(src_point.x > 0 && src_point.y > 0 && src_point.x < srcWidth - 1 && src_point.y < srcHeight - 1) {
        dst_x.at<float>(y, x) = src_point.x;
        dst_y.at<float>(y, x) = src_point.y;
        transformed_src.at<Vec3b>(y, x) = interpolate(from, src_point);
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        dst_x.at<float>(y, x) = dstWidth + 1;
        dst_y.at<float>(y, x) = dstHeight + 1;
        transformed_src.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
      }
    }
  }

  Mat final_image = to.clone();
  transformed_src.copyTo(final_image, construct_roi_mask(dstHeight, dstWidth, points));
  return final_image;
}

int main(int argc, char** argv )
{
  if (argc != 3) {
    printf("usage: ApplyHomography <Destination Image> <Source Image>\n");
    return -1;
  }

  Mat dst_image = imread(argv[1], 1);
  Mat src_image = imread(argv[2], 1);

  vector<Point2f> dst_points, src_points;
  receivePointCorrespondence(dst_points, "destination");
  receivePointCorrespondence(src_points, "source");

  Mat homography = computeHomography(dst_points, src_points);

  Mat modified_image = copyPerspectively(dst_image, src_image, homography, dst_points);
  write(argv[1], modified_image);
  return 0;
}
