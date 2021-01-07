#include <mvg.h>
#include <opencv2/imgproc.hpp>
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

// OpenCV harris for comparison of results
// Both algos give exactly the same results
void opencv_harris(Mat image) {
  Mat grayscale_img, normalized;
  float k = 0.05;
  float R = 151;
  cvtColor(image, grayscale_img, COLOR_RGB2GRAY);
  Mat result = Mat::zeros(grayscale_img.size(), CV_32FC1);
  cornerHarris(grayscale_img, result, 2, 3, k);
  normalize(result, normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  vector<KeyPoint> corners;
  for(int i = 0; i < normalized.rows; i++) {
    for(int j = 0; j < normalized.cols; j++) {
      if(normalized.at<float>(i,j) > R) {
        corners.push_back(KeyPoint(Point2f(j, i), 1.f));
      }
    }
  }
  cout << corners.size() << endl;
  drawKeypoints(image, corners, result, Scalar(0));
  display(result);
}

Mat non_maximal_suppression(Mat values) {
  Mat local_maximas = values.clone();

  int size = 20;

  for(int r = 0; r < local_maximas.rows - size; r += size) {
    for(int c = 0; c < local_maximas.cols - size; c += size) {
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

vector<KeyPoint> harris_stephens_corners(Mat image, float k, float threshold) {
  assert(image.channels() == 1);

  Mat Ix = convolve(image, SOBEL_X, CV_32FC1);
  Mat Iy = convolve(image, SOBEL_Y, CV_32FC1);

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
  Mat R = Mat(image.size(), CV_32FC1);

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

  vector<KeyPoint> corners;
  for(int i = 0; i < R_normalized.rows; i++) {
    for(int j = 0; j < R_normalized.cols; j++) {
      if(R_normalized.at<float>(i,j) > threshold) {
        corners.push_back(KeyPoint(Point2f(j, i), 1.f));
      }
    }
  }

  return corners;
}

Mat create_composite_image(Mat image1, Mat image2) {
  assert(image1.rows == image2.rows && image1.cols == image2.cols);
  Mat composite;
  hconcat(image1, image2, composite);
  return composite;
}

// TODO: handle case where points are at the borders of the image
float ssd(Mat image1, Point2f point1, Mat image2, Point2f point2) {
  assert(image1.channels() == 1 && image2.channels() == 1);

  // Patch size = half_patch_size * 2 + 1
  int half_patch_size = 10;
  float result = 0;

  for(int i = -half_patch_size; i <= half_patch_size; i++) {
    for(int j = -half_patch_size; j <= half_patch_size; j++) {
      float diff = (image1.at<char>(point1.x + i, point1.y + j) -
          image1.at<char>(point2.x + i, point2.y + j));
      result += diff * diff;
    }
  }

  return result;
}

bool sort_by_score(
    tuple <float, KeyPoint, KeyPoint> &a,
    tuple <float, KeyPoint, KeyPoint> &b) {
  return get<0>(a) < get<0>(b);
}

vector<tuple<KeyPoint, KeyPoint>> get_matches(
    Mat image1,
    vector<KeyPoint> corners1,
    Mat image2,
    vector<KeyPoint> corners2,
    float(*distance)(Mat, Point2f, Mat, Point2f)) {
  vector<tuple<float, KeyPoint, KeyPoint>> matches;

  for(auto it1 = corners1.begin(); it1 < corners1.end(); it1++) {
    float min_distance = numeric_limits<float>::infinity();
    tuple<float, KeyPoint, KeyPoint> best_pair;
    auto best_position = corners2.begin();
    for(auto it2 = corners2.begin(); it2 < corners2.end(); it2++) {
      float ssd = distance(image1, (*it1).pt, image2, (*it2).pt);
      if(ssd < min_distance) {
        min_distance = ssd;
        best_pair = make_tuple(min_distance, *it1, *it2);
        best_position = it2;
      }
    }
    corners2.erase(best_position);
    matches.push_back(best_pair);
  }
  sort(matches.begin(), matches.end(), sort_by_score);

  int i = 0;
  int num_matches = 50;
  vector<tuple<KeyPoint, KeyPoint>> top_matches;
  for(auto it = matches.begin(); it < matches.end() && i < num_matches; it++, i++) {
    top_matches.push_back(make_tuple(get<1>(*it), get<2>(*it)));
  }
  return top_matches;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("usage: match_features <Image1> <Image2> <k> <threshold>\n");
    return -1;
  }

  Mat image1 = imread(argv[1], 1);
  Mat image2 = imread(argv[2], 1);
  Mat grayscale_img1, grayscale_img2;
  cvtColor(image1, grayscale_img1, COLOR_RGB2GRAY);
  cvtColor(image2, grayscale_img2, COLOR_RGB2GRAY);

  float k = atof(argv[3]);
  float threshold = atof(argv[4]);
  vector<KeyPoint> corners1 = harris_stephens_corners(grayscale_img1, k, threshold);
  vector<KeyPoint> corners2 = harris_stephens_corners(grayscale_img2, k, threshold);

  Mat result;
  drawKeypoints(image1, corners1, result, Scalar(50));
  write(argv[1], result, "-harris-corners");
  drawKeypoints(image2, corners2, result, Scalar(50));
  write(argv[2], result, "-harris-corners");

  // Draw line between matches
  Mat composite_image = create_composite_image(image1, image2);
  vector<tuple<KeyPoint, KeyPoint>> matches = get_matches(grayscale_img1, corners1, grayscale_img2, corners2, &ssd);

  Point2f offset = Point2f(image1.cols, 0);
  for(auto it = matches.begin(); it < matches.end(); it++) {
    KeyPoint kp1 = get<0>(*it);
    KeyPoint kp2 = get<1>(*it);
    line(composite_image, kp1.pt, kp2.pt + offset, Scalar(255), 2, LINE_8);
  }

  display(composite_image);

  return 0;
}
