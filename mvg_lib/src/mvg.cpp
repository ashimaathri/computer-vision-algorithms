#include "mvg.h"
#include <cstring>
using namespace cv;
using namespace std;

const Mat SOBEL_X = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const Mat SOBEL_Y = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

// Assuming kernel is odd and square
Mat convolve(Mat image, Mat kernel, int type) {
  assert(image.channels() == 1);

  int kernel_size = kernel.size().width;
  int half_kernel_size = kernel_size / 2;

  Mat padded;

  copyMakeBorder(
      image,
      padded,
      half_kernel_size,
      half_kernel_size,
      half_kernel_size,
      half_kernel_size,
      BORDER_DEFAULT,
      0);

  int height = padded.size().height;
  int width = padded.size().width;
  int result_height = height - half_kernel_size * 2;
  int result_width = width - half_kernel_size * 2;

  Mat result = Mat(result_height, result_width, type, Scalar(0));;

  for(int r = half_kernel_size; r <= result_height; r++) {
    for(int c = half_kernel_size; c <= result_width; c++) {
      double value = 0;
      for(int k_r = 0; k_r < kernel_size; k_r++) {
        for(int k_c = 0; k_c < kernel_size; k_c++) {
          value += (
              padded.at<uint8_t>(r + half_kernel_size - k_r, c + half_kernel_size - k_c) *
              kernel.at<float>(k_r, k_c));
        }
      }
      switch(type) {
        case CV_32FC1:
          result.at<float>(r - half_kernel_size, c - half_kernel_size) = value;
          break;
        case CV_64FC1:
          result.at<double>(r - half_kernel_size, c - half_kernel_size) = value;
          break;
        default:
          result.at<uint8_t>(r - half_kernel_size, c - half_kernel_size) = value;
      }
    }
  }

  return result;
}

Mat convolve_color(Mat image, Mat kernel) {
  assert(image.channels() == 3);

  Mat result, src_channels[3];
  vector<Mat> dst_channels;

  split(image, src_channels);

  for(int i = 0; i < 3; i++) {
    Mat result = convolve(src_channels[i], kernel, src_channels[i].type());
    dst_channels.push_back(result);
  }

  merge(dst_channels, result);

  return result;
}

Mat construct_color_image() {
  Mat channel = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);

  vector<Mat> channels;
  for(int i = 0; i < 3; i++) {
    channels.push_back(channel);
  }

  Mat result;
  merge(channels, result);

  return result;
}

void print_color_image(Mat image) {
  assert(image.channels() == 3);

  Mat channels[3];
  split(image, channels);

  for(int i = 0; i < 3; i++) {
    cout << channels[i] << endl;
  }
}

Mat non_maximal_suppression(Mat values) {
  Mat local_maximas = values.clone();

  int size = 10;

  for(int r = 0; r < local_maximas.rows - size; r++) {
    for(int c = 0; c < local_maximas.cols - size; c++) {
      float max_value = -numeric_limits<float>::infinity();
      int r_offset, c_offset;
      for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
          float current = local_maximas.at<float>(r + i, c + j);
          if(current > max_value) {
            max_value = current;
            r_offset = i;
            c_offset = j;
          }
        }
      }
      Mat patch = local_maximas(Rect(c, r, size, size));
      patch.setTo(Scalar::all(0));
      patch.at<float>(r_offset, c_offset) = max_value;
    }
  }

  return local_maximas;
}

vector<uint8_t> makeDescriptor(Mat image, Point2f point) {
  assert(image.channels() == 3);

  int numElements = 0;
  int windowSize = 7;
  vector<uint8_t> descriptor(image.channels() * windowSize * windowSize, 0);

  int halfWindowSize = windowSize / 2;
  Vec3b centralIntensity = image.at<Vec3b>(point.y, point.x);

  for(int i = -halfWindowSize; i <= halfWindowSize; i++) {
    for(int j = -halfWindowSize; j <= halfWindowSize; j++) {
      int y = point.y + i;
      int x = point.x + j;
      if(y >= image.rows) {
        y = image.rows - 1;
      }
      if(x >= image.cols) {
        x = image.cols - 1;
      }
      Vec3b intensity = image.at<Vec3b>(y, x);
      for(int c = 0; c < 3; c++) {
        descriptor.push_back(intensity[c]);
      }
    }
  }

  return descriptor;
}

vector<tuple<KeyPoint, vector<uint8_t>>> harris_stephens_corners(Mat image, float k, float threshold) {
  Mat grayscale;

  cvtColor(image, grayscale, COLOR_RGB2GRAY);

  Mat Ix = convolve(grayscale, SOBEL_X, CV_32FC1);
  Mat Iy = convolve(grayscale, SOBEL_Y, CV_32FC1);

  Mat Ixx, Iyy, Ixy;

  multiply(Ix, Ix, Ixx);
  multiply(Ix, Iy, Ixy);
  multiply(Iy, Iy, Iyy);

  Size smoothing_size = Size(3, 3);

  boxFilter(Ixx, Ixx, Ixx.depth(), smoothing_size, Point(-1, -1), true,
      BORDER_DEFAULT);
  boxFilter(Ixy, Ixy, Ixy.depth(), smoothing_size, Point(-1, -1), true,
      BORDER_DEFAULT);
  boxFilter(Iyy, Iyy, Iyy.depth(), smoothing_size, Point(-1, -1), true,
      BORDER_DEFAULT);

  Size size = Ixx.size();
  Mat R = Mat(grayscale.size(), CV_32FC1);

  for(int r = 0; r < size.height; r++) {
    for(int c = 0; c < size.width; c++) {
      float M11 = Ixx.at<float>(r, c);
      float M12 = Ixy.at<float>(r, c);
      float M22 = Iyy.at<float>(r, c);
      float determinant = M11 * M22 - M12 * M12;
      float trace = M11 + M22;
      R.at<float>(r, c) = determinant - (k * trace * trace);
    }
  }

  Mat R_normalized;
  normalize(R, R_normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  R_normalized = non_maximal_suppression(R_normalized);

  vector<tuple<KeyPoint, vector<uint8_t>>> corners;
  for(int i = 0; i < R_normalized.rows; i++) {
    for(int j = 0; j < R_normalized.cols; j++) {
      if(R_normalized.at<float>(i,j) > threshold) {
        Point2f point = Point2f(j, i);
        corners.push_back(make_tuple(KeyPoint(point, 1.f), makeDescriptor(image, point)));
      }
    }
  }

  cout << "Found " << corners.size() << " corners" << endl;
  return corners;
}

Mat create_composite_image(Mat image1, Mat image2) {
  assert(image1.rows == image2.rows && image1.cols == image2.cols);
  Mat composite;
  hconcat(image1, image2, composite);
  return composite;
}

float sumAbsDiff(vector<uint8_t> a, vector<uint8_t> b) {
  assert(a.size() == b.size());

  float result = 0;

  for(int i = 0; i < a.size(); i++) {
    result += abs(a[i] - b[i]);
  }

  return result;
}

bool sortByScore(
    tuple <int, int, float> &a,
    tuple <int, int, float> &b) {
  return get<2>(a) < get<2>(b);
}

vector<tuple<KeyPoint, KeyPoint>> getMatches(
    vector<tuple<KeyPoint, vector<uint8_t>>> corners1,
    vector<tuple<KeyPoint, vector<uint8_t>>> corners2) {
  bool swapped = false;
  vector<tuple<KeyPoint, vector<uint8_t>>> keyCorners, matchCorners; 

  if(corners1.size() < corners2.size()) {
    keyCorners = corners1;
    matchCorners = corners2;
  } else {
    swapped = true;
    keyCorners = corners2;
    matchCorners = corners1;
  }

  map<int, tuple<int, float>> matches;
  vector<bool> visited(keyCorners.size(), false);

  for(int i = 0; i < keyCorners.size(); i++) {
    if(visited.at(i)) {
      continue;
    }
    int bestPosition = 0;
    float bestScore = numeric_limits<float>::infinity();

    tuple<KeyPoint, vector<uint8_t>> keyCorner = keyCorners.at(i);

    for(int j = 0; j < matchCorners.size(); j++) {
      float score = sumAbsDiff(get<1>(keyCorner), get<1>(matchCorners.at(j)));

      if(score < bestScore) {
        bool matchNotTaken = (matches.find(j) == matches.end());
        if(matchNotTaken || get<1>(matches[j]) > score) {
          bestScore = score;
          bestPosition = j;
        }
      }
    }

    visited.at(i) = true;

    bool matchNotTaken = (matches.find(bestPosition) == matches.end());

    if(matchNotTaken) {
      matches[bestPosition] = make_tuple(i, bestScore);
    } else {
      int next_i = get<0>(matches[bestPosition]);
      matches[bestPosition] = make_tuple(i, bestScore);
      i = next_i;
      visited.at(i) = false;
    }
  }
  vector<tuple<int, int, float>> sortedMatches;

  for(auto it = matches.begin(); it != matches.end(); it++) {
    sortedMatches.push_back(make_tuple(it->first, get<0>(it->second), get<1>(it->second)));
  }

  sort(sortedMatches.begin(), sortedMatches.end(), sortByScore);

  int i = 0;
  int numMatches = 30;
  vector<tuple<KeyPoint, KeyPoint>> topMatches;

  if(swapped) {
    for(auto it = sortedMatches.begin(); it < sortedMatches.end() && i < numMatches; it++, i++) {
      topMatches.push_back(make_tuple(get<0>(matchCorners.at(get<0>(*it))), get<0>(keyCorners.at(get<1>(*it)))));
    }
  } else {
    for(auto it = sortedMatches.begin(); it < sortedMatches.end() && i < numMatches; it++, i++) {
      topMatches.push_back(make_tuple(get<0>(keyCorners.at(get<1>(*it))), get<0>(matchCorners.at(get<0>(*it)))));
    }
  }

  return topMatches;
}

Mat solveConicSystem(vector<Mat> points) {
  int numLines = points.size();

  assert(numLines == 5);
  const int numRows = 5;
  const int numCols = 6;

  float linearSystem[numRows][numCols];

  int sizeOfRow = numCols * sizeof(float);

  for(int i = 0; i < 5; i++) {
    float x1 = points[i].at<float>(0, 0);
    float x2 = points[i].at<float>(0, 1);
    float x3 = points[i].at<float>(0, 2);
    const float row[numCols] = {x1 * x1, x1 * x2, x2 * x2, x1, x2, 1};
    memcpy(linearSystem + i, (void*)&row, sizeOfRow);
  }

  Mat coefficients = getNullSpace((void*)&linearSystem, numRows, numCols);
  float a = coefficients.at<float>(0, 0);
  float b = coefficients.at<float>(0, 1) / 2;
  float c = coefficients.at<float>(0, 2);
  float d = coefficients.at<float>(0, 3) / 2;
  float e = coefficients.at<float>(0, 4) / 2;
  float f = coefficients.at<float>(0, 5);
  Mat conic = (Mat_<float>(3, 3) << a, b, d, b, c, e, d, e, f);
  // Make sure to return positive-definite conic
  return (a > 0 ? conic : -conic);
}

Mat getConic() {
  vector<Mat> points;

  cout << "Enter the 5 points on the imaged circle" << endl;
  for(int i = 0; i < 5; i++) {
    points.push_back(normalize(readHomogeneousCoordinates()));
  }

  return solveConicSystem(points);
}

Mat intersectLineAndConic(Mat conic, Mat P, Mat Q) {
  Mat Pt, Qt;

  transpose(P, Pt);
  transpose(Q, Qt);

  Mat A = Pt * conic * P;
  Mat B = 2 * Pt * conic * Q;
  Mat C = Qt * conic * Q;

  float a = A.at<float>(0, 0);
  float b = B.at<float>(0, 0);
  float c = C.at<float>(0, 0);

  float alpha = -b / (2 * a);
  float beta = sqrt(4 * a * c - b * b) / (2 * a);

  Mat circularPoint;

  Mat channels[2] = {alpha * P + Q, beta * P};
  merge(channels, 2, circularPoint);

  return circularPoint;
}

Mat transformConic(Mat pointTransform, Mat conic) {
  Mat pointTransform_t;

  transpose(pointTransform, pointTransform_t);

  return pointTransform_t.inv() * conic * pointTransform.inv();
}

Mat getConjugate(Mat point) {
  Mat conjugate;
  vector<Mat> channels(2);

  split(point, channels);
  channels = {channels[0], -channels[1]};
  merge(channels, conjugate);

  return conjugate;
}

Mat complexMatMul(Mat matrix1, Mat matrix2) {
  Mat m1, m2;

  int n = matrix1.rows;
  int m = matrix2.cols;
  int numChannels1 = matrix1.channels();
  int numChannels2 = matrix2.channels();
  assert(matrix1.cols == matrix2.rows);
  assert(numChannels1 <= 2 && numChannels2 <= 2);

  if(numChannels1 < 2) {
    Mat tmp[] = {matrix1, Mat::zeros(matrix1.size(), CV_32FC1)};
    merge(tmp, 2, m1);
    m2 = matrix2;
  }

  if(numChannels2 < 2) {
    Mat tmp[] = {matrix2, Mat::zeros(matrix2.size(), CV_32FC1)};
    merge(tmp, 2, m2);
    m1 = matrix1;
  }

  Mat product(n, m, CV_32FC2);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      Mat result, row, column;
      row = m1.row(i);
      transpose(m2.col(j), column);
      mulSpectrums(row, column, result, 0);
      Scalar resultSum = sum(result);
      product.at<Vec2f>(i, j) = Vec2f(resultSum[0], resultSum[1]);
    }
  }

  return product;
}

Mat fullDecomp(Mat symmetricMat) {
  Mat W, U, Ut, W_sqrt, diagonal;
  cv::SVDecomp(symmetricMat, W, U, Ut, cv::SVD::FULL_UV);
  cv::sqrt(W, W_sqrt);
  return U * diagonal.diag(W_sqrt) * Ut;
}

Mat choleskyUpperDecomp(Mat matrix) {
  Mat K, Q, W, U, Ut, W_sqrt, diagonal;

  cv::SVDecomp(matrix, W, U, Ut, cv::SVD::FULL_UV);
  cv::sqrt(W, W_sqrt);
  Mat V = U * diagonal.diag(W_sqrt);
  // We don't have a function to perform RQ decomposition for 2x2 matrix so we
  // convert the 2x2 into a 3x3 such that the result for the 2x2 is the same as
  // it would be for the corresponding 3x3.
  Mat V_3x3 = (Mat_<float>(3, 3) << V.at<float>(0, 0), V.at<float>(0, 1),
      0, V.at<float>(1, 0), V.at<float>(1, 1), 0, 0, 0, 1);
  RQDecomp3x3(V_3x3, K, Q);

  return (Mat_<float>(2, 2) << K.at<float>(0, 0), K.at<float>(0, 1),
      K.at<float>(1, 0), K.at<float>(1, 1));
}

Mat choleskyDecomposition(Mat matrix) {
  Mat upperTriangular = matrix.clone();

  cv::Cholesky((float*)upperTriangular.ptr(), upperTriangular.step,
      upperTriangular.rows, NULL, 0, 0);

  upperTriangular = upperTriangular.t();

  for(int i = 1; i < upperTriangular.rows; i++) {
    for(int j = 0; j < i; j++) {
      upperTriangular.at<float>(i, j) = 0;
    }
  }

  Mat lowerTriangular;
  cv::transpose(upperTriangular, lowerTriangular);
  return lowerTriangular;
}

Mat normalize(Mat coordinates) {
  return coordinates / coordinates.at<float>(2, 0);
}

void printVector(vector <float> v) {
  cout << "Printing vector" << endl;
  for(std::vector<float>::const_iterator i = v.begin(); i != v.end(); ++i) {
    cout << *i << ' ';
  }
  cout << endl;
}

Mat getNullSpace(void* linearSystem, int numRows, int numCols, int finalRows) {
  Mat W, U, Vt;
  cv::SVDecomp(
      Mat(numRows, numCols, CV_32FC1, linearSystem),
      W,
      U,
      Vt,
      cv::SVD::FULL_UV);
  return Vt(Range(numRows, numRows + 1), Range::all()).reshape(0, finalRows);
}

Mat readHomogeneousCoordinates() {
  float x1, x2, x3;
  std::cin >> x1;
  std::cin >> x2;
  std::cin >> x3;
  return (Mat_<float>(3, 1) << x1, x2, x3);
}

Mat computeHomography(std::vector <Point2f> from, std::vector <Point2f> to) {
  int numPoints = from.size();

  bool isMinimal = (numPoints / 4) > 0;
  bool areNumCorrespondencesEqual = (to.size() == numPoints);
  assert(isMinimal && areNumCorrespondencesEqual);

  int numCols = 9;
  int numRows = numPoints * 2;
  Mat result;

  for(auto it_from = from.begin(), it_to = to.begin();
      it_from < from.end();
      it_from++, it_to++) {
    Mat row;

    row = (Mat_<float>(1, numCols) << -(*it_from).x, -(*it_from).y, -1,
        0, 0, 0, (*it_from).x * (*it_to).x, (*it_from).y * (*it_to).x,
        (*it_to).x);
    result.push_back(row);

    row = (Mat_<float>(1, numCols) << 0, 0, 0, -(*it_from).x, -(*it_from).y,
        -1, (*it_from).x * (*it_to).y, (*it_from).y * (*it_to).y, (*it_to).y);
    result.push_back(row);
  }

  Mat W, U, Vt;
  cv::SVDecomp(
      result,
      W,
      U,
      Vt,
      cv::SVD::FULL_UV);
  return Vt(Range(8, 9), Range::all()).reshape(0, 3);
}

Point2f transform_point(Mat homography, float x, float y) {
  Point2f transformed_point;

  Mat homogeneous_src_point = (Mat_<float>(3, 1) << x, y, 1);
  Mat homogeneous_dst_point = homography * homogeneous_src_point;

  float w = homogeneous_dst_point.at<float>(2, 0);
  transformed_point.x = homogeneous_dst_point.at<float>(0, 0) / w;
  transformed_point.y = homogeneous_dst_point.at<float>(1, 0) / w;

  return transformed_point;
}

// Very slow because it goes pixel by pixel instead of using GPU matrix
// operations
Vec3b interpolate(Mat image, Point2f point) {
  float int_x, int_y;
  float frac_x = modf(point.x, &int_x);
  float frac_y = modf(point.y, &int_y);

  Vec3b f11 = image.at<Vec3b>(int_y, int_x);
  Vec3b f12 = image.at<Vec3b>(int_y + 1, int_x);
  Vec3b f21 = image.at<Vec3b>(int_y, int_x + 1);
  Vec3b f22 = image.at<Vec3b>(int_y + 1, int_x + 1);
  Vec3b interpolated_value;

  Mat x_weights = (Mat_<float>(1, 2) << (1 - frac_x), frac_x);
  Mat y_weights = (Mat_<float>(2, 1) << (1 - frac_y), frac_y);

  for(int i = 0; i < image.channels(); i++) {
    Mat neighbor_intensities = (Mat_<float>(2, 2) << f11[i], f12[i], f21[i], f22[i]);
    Mat result = x_weights * neighbor_intensities * y_weights;
    interpolated_value[i] = result.at<float>(0, 0);
  }

  return interpolated_value;
}

// Back project the corners of the image to get the bounds for the final result
// image, it's not simply image.size()
// Pass dst size as argument?
Mat changePerspective(Mat image, Mat homography) {
  cout << "Applying homography" << endl;
  // Origin of our coordinate system is top-left of image
  int height = image.size().height;
  int width = image.size().width;

  Mat dstX = Mat(height, width, CV_32FC1);
  Mat dstY = Mat(height, width, CV_32FC1);
  Mat transformedSrc = Mat(height, width, image.type());

  for(int x = 0; x < width; x++) {
    for(int y = 0; y < height; y++) {
      Point2f srcPoint = transform_point(homography, x, y);
      if(srcPoint.x > 0 && srcPoint.y > 0 && srcPoint.x < width - 1 && srcPoint.y < height - 1) {
        dstX.at<float>(y, x) = srcPoint.x;
        dstY.at<float>(y, x) = srcPoint.y;
        transformedSrc.at<Vec3b>(y, x) = interpolate(image, srcPoint);
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        dstX.at<float>(y, x) = width + 1;
        dstY.at<float>(y, x) = height + 1;
        transformedSrc.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
      }
    }
  }
  return transformedSrc;
}

Mat computeScaling(Mat image, Mat homography) {
  float width = image.size().width;
  float height = image.size().height;

  const float extent[3][4] = {
    {0, width, 0, width},
    {0, 0, height, height},
    {1, 1, 1, 1},
  };

  Mat worldExtent = homography.inv() * Mat(3, 4, CV_32FC1, (void*)&extent);

  float minWidth = std::numeric_limits<float>::infinity();
  float maxWidth = -std::numeric_limits<float>::infinity();
  float minHeight = std::numeric_limits<float>::infinity();
  float maxHeight = -std::numeric_limits<float>::infinity();

  for(int i = 0; i < 4; i++) {
    float z = worldExtent.at<float>(2, i);
    float x = worldExtent.at<float>(0, i) / z;
    float y = worldExtent.at<float>(1, i) / z;
    minWidth =  x < minWidth ? x : minWidth;
    maxWidth =  x > maxWidth ? x : maxWidth;
    minHeight =  y < minHeight ? y : minHeight;
    maxHeight =  y > maxHeight ? y : maxHeight;
  }

  float widthScaleFactor = (maxWidth - minWidth)/width;
  float heightScaleFactor = (maxHeight - minHeight)/height;

  float finalScaleFactor = (widthScaleFactor > heightScaleFactor ?  widthScaleFactor : heightScaleFactor);

  // Scale the smaller image up and translate to the bottom left to match with actual world extent
  return (Mat_<float>(3, 3) << finalScaleFactor, 0, minWidth, 0, finalScaleFactor, minHeight, 0, 0, 1);
}

void write(char* nameOfOriginal, Mat image, std::string suffix) {
  string path = string(nameOfOriginal);
  string filename = path.substr(path.find_last_of("/") + 1, path.length());
  string imagename = filename.substr(0, filename.find_last_of("."));
  string outputFilename = "output/" + imagename + suffix + ".jpg";
  cout << "Writing to " << outputFilename << endl;
  imwrite(outputFilename, image);
}

void display(Mat image) {
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  image.convertTo(image, CV_8U);
  imshow("Display Image", image);
  waitKey(0);
}

void receivePointCorrespondence(std::vector <Point2f> &points, const std::string &type) {
  std::cout << "Enter the x,y coordinates of four points in the " << type << " image\n";
  for(int i = 0; i < 4; i++) {
    float x, y;
    std::cin >> x;
    std::cin >> y;
    points.push_back(Point2f(x, y));
  }
}
