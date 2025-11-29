#pragma once

#include <vector>
#include "kidnapped_vehicle/helper_functions.h"

struct Particle {
  int id;
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};

class ParticleFilter {
public:
  ParticleFilter() : num_particles(0), is_initialized(false) {}
  ~ParticleFilter() {}

  // Initialize with Gaussian noise around initial estimate (x, y, theta)
  void init(double x, double y, double theta, double std[]);

  // Predict particle states given control input
  void prediction(double delta_t, double std_pos[],
                  double velocity, double yaw_rate);

  // Associate observations with predicted landmarks
  void dataAssociation(std::vector<LandmarkObs> predicted,
                       std::vector<LandmarkObs>& observations);

  // Update particle weights using sensor measurements
  void updateWeights(double sensor_range, double std_landmark[],
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);

  // Resample particles according to their weights
  void resample();

  // Check if initialized
  bool initialized() const { return is_initialized; }

  std::vector<Particle> particles;
  std::vector<double> weights;

private:
  int num_particles = 100;
  bool is_initialized;
};

