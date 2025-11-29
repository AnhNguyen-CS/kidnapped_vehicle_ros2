#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

// Observation in vehicle or map coordinates
struct LandmarkObs {
  int id;      // Id of matching landmark
  double x;    // x position
  double y;    // y position
};

// Simple map structure with a list of landmarks
class Map {
public:
  struct single_landmark_s {
    int id_i;   // Landmark ID
    float x_f;  // Landmark x position in map coordinates [m]
    float y_f;  // Landmark y position in map coordinates [m]
  };

  std::vector<single_landmark_s> landmark_list;
};

// Euclidean distance helper
inline double dist(double x1, double y1, double x2, double y2) {
  return std::sqrt((x1 - x2) * (x1 - x2) +
                   (y1 - y2) * (y1 - y2));
}

// Normalize angle to [-pi, pi]
inline double normalize_angle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

// Read map data from a file (same format as Udacity map_data.txt)
inline bool read_map_data(const std::string &filename, Map &map) {
  std::ifstream in_file(filename.c_str(), std::ifstream::in);
  if (!in_file) {
    return false;
  }

  map.landmark_list.clear();

  float x;
  float y;
  int id;

  while (in_file >> x >> y >> id) {
    Map::single_landmark_s lm;
    lm.id_i = id;
    lm.x_f = x;
    lm.y_f = y;
    map.landmark_list.push_back(lm);
  }

  return true;
}

