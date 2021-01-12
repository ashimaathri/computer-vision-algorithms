#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

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
    return -numeric_limits<int>::infinity();
  }
}

float transferError(KeyPoint src, KeyPoint dst, Mat homography) {
  Mat srcPoint = (Mat_<float>(3, 1) << src.pt.x, src.pt.y, 1);
  Mat dstPoint = (Mat_<float>(3, 1) << dst.pt.x, dst.pt.y, 1);
  Mat transformedSrcPoint = homography * srcPoint;
  Mat invertedHomography;
  invert(homography, invertedHomography);
  Mat transformedDstPoint = invertedHomography * dstPoint;

  return norm(srcPoint - transformedSrcPoint) + norm(dstPoint - transformedDstPoint);
}

vector<tuple<KeyPoint, KeyPoint>> getInliers(
    Mat homography,
    vector<tuple<KeyPoint, KeyPoint>> matches) {

  float measurement_error_sigma = 1;
  float threshold = 5.99 * measurement_error_sigma;
  vector<tuple<KeyPoint, KeyPoint>> inliers;

  for(auto it = matches.begin(); it < matches.end(); it++) {
    KeyPoint src = get<0>(*it);
    KeyPoint dst = get<1>(*it);
    if(transferError(src, dst, homography) < threshold) {
      inliers.push_back((*it));
    }
  }

  return inliers;
}

Mat ransac(Mat srcImage, Mat dstImage, float k, float threshold) {
  Mat homography;
  Mat srcGrayscale, dstGrayscale;

  cvtColor(srcImage, srcGrayscale, COLOR_RGB2GRAY);
  cvtColor(dstImage, dstGrayscale, COLOR_RGB2GRAY);

  vector<KeyPoint> srcCorners = harris_stephens_corners(srcGrayscale, k, threshold);
  vector<KeyPoint> dstCorners = harris_stephens_corners(dstGrayscale, k, threshold);

  vector<tuple<KeyPoint, KeyPoint>> matches = get_matches(srcGrayscale,
      srcCorners, dstGrayscale, dstCorners, &ssd);

  int i = 0;
  int sampleSize = 4;
  int numCorners = matches.size();
  int bestNumInliers = 0;
  vector<tuple<KeyPoint, KeyPoint>> bestInliers;
  int numIterations = -numeric_limits<int>::infinity();

  while(true) {
    vector<Point2f> srcSample, dstSample;

    for(int i = 0; i < sampleSize; i++) {
      srcSample.push_back(srcCorners.at(rand() % numCorners).pt);
      dstSample.push_back(dstCorners.at(rand() % numCorners).pt);
    }

    Mat homography = computeHomography(srcSample, dstSample);

    vector<tuple<KeyPoint, KeyPoint>> inliers = getInliers(homography, matches);

    if(inliers.size() > bestNumInliers) {
      bestNumInliers = inliers.size();
      bestInliers = inliers;
    }

    if(++i > getNumIterations((numCorners - inliers.size()) / numCorners)) {
      break;
    }
  }
  vector<Point2f> inlierSrcCorners, inlierDstCorners;
  for(auto it = bestInliers.begin(); it < bestInliers.end(); it++) {
    inlierSrcCorners.push_back(get<0>(*it).pt);
    inlierDstCorners.push_back(get<1>(*it).pt);
  }
  return computeHomography(inlierSrcCorners, inlierDstCorners);
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

  float k = atof(argv[3]);
  float threshold = atof(argv[4]);
  vector<Mat> homographies;

  for(int i = 0; i < numImages; i++) {
    if(i == referenceImageIndex) {
      homographies.push_back(Mat::eye(3, 3, CV_32FC1));
    } else {
      homographies.push_back(ransac(images.at(i), referenceImage, k, threshold));
    }
  }

  return 0;
}
