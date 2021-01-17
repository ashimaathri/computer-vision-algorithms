#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

// Based on CDF of Chi-Square distribution
int getNumIterations(float outlierFraction) {
  if(outlierFraction <= .05) {
    return 3;
  } else if(outlierFraction > .05 && outlierFraction <= .1) {
    return 5;
  } else if(outlierFraction > .1 && outlierFraction <= .2) {
    return 9;
  } else if(outlierFraction > .2 && outlierFraction <= .25) {
    return 13;
  } else if(outlierFraction > .25 && outlierFraction <= .3) {
    return 17;
  } else if(outlierFraction > .3 && outlierFraction <= .4) {
    return 34;
  } else if(outlierFraction > .4 && outlierFraction <= .5) {
    return 72;
  } else {
    return INT_MAX;
  }
}

Mat normalizeHomogeneousPoint(Mat point) {
  return point / point.at<float>(2, 0);
}

float transferError(Point2f src, Point2f dst, Mat homography) {
  Mat srcPoint = (Mat_<float>(3, 1) << src.x, src.y, 1);
  Mat dstPoint = (Mat_<float>(3, 1) << dst.x, dst.y, 1);
  Mat invertedHomography;
  invert(homography, invertedHomography);
  Mat transformedSrcPoint = invertedHomography * srcPoint;
  Mat transformedDstPoint = homography * dstPoint;

  return (norm(dstPoint - normalizeHomogeneousPoint(transformedSrcPoint)) +
      norm(srcPoint - normalizeHomogeneousPoint(transformedDstPoint)));
}

vector<tuple<KeyPoint, KeyPoint>> getInliers(
    Mat homography,
    vector<tuple<KeyPoint, KeyPoint>> matches) {

  float measurement_error_sigma = 0.5;
  float threshold = 5.99 * measurement_error_sigma;
  float threshold_square = threshold * threshold;

  vector<tuple<KeyPoint, KeyPoint>> inliers;

  for(auto it = matches.begin(); it < matches.end(); it++) {
    float error = transferError(get<0>(*it).pt, get<1>(*it).pt, homography);
    if(error < threshold_square) {
      inliers.push_back((*it));
    }
  }

  return inliers;
}

void drawMatches(Mat image1, Mat image2, vector<tuple<KeyPoint, KeyPoint>> matches) {
  Mat composite_image = create_composite_image(image1, image2);
  Point2f offset = Point2f(image1.cols, 0);
  for(auto it = matches.begin(); it < matches.end(); it++) {
    KeyPoint kp1 = get<0>(*it);
    KeyPoint kp2 = get<1>(*it);
    line(composite_image, kp1.pt, kp2.pt + offset, Scalar(255), 2, LINE_8);
  }
  display(composite_image);
}

Mat ransac(Mat srcImage, Mat dstImage, float k, float threshold) {
  Mat homography;
  Mat srcGrayscale, dstGrayscale;

  cvtColor(srcImage, srcGrayscale, COLOR_RGB2GRAY);
  cvtColor(dstImage, dstGrayscale, COLOR_RGB2GRAY);

  vector<KeyPoint> srcCorners = harris_stephens_corners(srcGrayscale, k, threshold);
  vector<KeyPoint> dstCorners = harris_stephens_corners(dstGrayscale, k, threshold);

  vector<tuple<KeyPoint, KeyPoint>> matches = getMatches(srcGrayscale,
      srcCorners, dstGrayscale, dstCorners, &ssd);

  int i = 0;
  int sampleSize = 4;
  int numCorners = matches.size();
  int bestNumInliers = 0;
  vector<tuple<KeyPoint, KeyPoint>> bestInliers;
  int numIterations = -numeric_limits<int>::infinity();

  while(true) {
    vector<Point2f> srcSample, dstSample;
    vector<tuple<KeyPoint, KeyPoint>> selectedMatches;

    for(int i = 0; i < sampleSize; i++) {
      tuple<KeyPoint, KeyPoint> match = matches.at(rand() % numCorners);
      srcSample.push_back(get<0>(match).pt);
      dstSample.push_back(get<1>(match).pt);
      selectedMatches.push_back(match);
    }
    drawMatches(srcImage, dstImage, selectedMatches);

    Mat homography = computeHomography(dstSample, srcSample);
    display(changePerspective(srcImage, homography));

    vector<tuple<KeyPoint, KeyPoint>> inliers = getInliers(homography, matches);

    if(inliers.size() > bestNumInliers) {
      bestNumInliers = inliers.size();
      bestInliers = inliers;
    }

    float outlierFraction = ((float)numCorners - (float)inliers.size()) / (float)numCorners;

    int numIterations = getNumIterations(outlierFraction);

    if(++i > numIterations) {
      break;
    }
  }
  vector<Point2f> inlierSrcCorners, inlierDstCorners;
  for(auto it = bestInliers.begin(); it < bestInliers.end(); it++) {
    inlierSrcCorners.push_back(get<0>(*it).pt);
    inlierDstCorners.push_back(get<1>(*it).pt);
  }
  return computeHomography(inlierDstCorners, inlierSrcCorners);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: panorama <number of images> <k> <threshold>\n");
    return -1;
  }

  int numImages = atoi(argv[1]);
  vector<Mat> images;

  for(int i = 0; i < numImages; i++) {
    string filename;
    cin >> filename;
    images.push_back(imread(filename));
  }

  int referenceImageIndex = numImages / 2;
  Mat referenceImage = images.at(referenceImageIndex);

  float k = atof(argv[2]);
  float threshold = atof(argv[3]);
  vector<Mat> homographies;

  for(int i = 0; i < numImages; i++) {
    if(i == referenceImageIndex) {
      homographies.push_back(Mat::eye(3, 3, CV_32FC1));
    } else {
      homographies.push_back(ransac(images.at(i), referenceImage, k, threshold));
      display(changePerspective(images.at(i), homographies.at(i)));
    }
  }

  return 0;
}
