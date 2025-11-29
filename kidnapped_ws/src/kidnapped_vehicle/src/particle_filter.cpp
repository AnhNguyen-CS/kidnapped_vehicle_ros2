#include "kidnapped_vehicle/particle_filter.h"
#include "kidnapped_vehicle/helper_functions.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <iterator>

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  if (is_initialized) {
    return;
  }

  num_particles = 2000;  // you can tune this: 50, 100, 200...

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.clear();
  weights.clear();

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle &p = particles[i];
    double theta = p.theta;

    double new_x, new_y, new_theta;

    if (fabs(yaw_rate) > 1e-5) {
      double v_over_yaw = velocity / yaw_rate;
      new_x = p.x + v_over_yaw *
                     (sin(theta + yaw_rate * delta_t) - sin(theta));
      new_y = p.y + v_over_yaw *
                     (cos(theta) - cos(theta + yaw_rate * delta_t));
      new_theta = theta + yaw_rate * delta_t;
    } else {
      new_x = p.x + velocity * delta_t * cos(theta);
      new_y = p.y + velocity * delta_t * sin(theta);
      new_theta = theta;
    }

    // Add gaussian noise
    p.x     = new_x + dist_x(gen);
    p.y     = new_y + dist_y(gen);
    p.theta = new_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  for (auto &obs : observations) {
    double min_dist = std::numeric_limits<double>::max();
    int best_id = -1;

    for (const auto &pred : predicted) {
      double d = dist(obs.x, obs.y, pred.x, pred.y);
      if (d < min_dist) {
        min_dist = d;
        best_id = pred.id;
      }
    }

    obs.id = best_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double sig_x2 = sig_x * sig_x;
  double sig_y2 = sig_y * sig_y;
  double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

  for (int i = 0; i < num_particles; ++i) {
    Particle &p = particles[i];

    // 1) Select landmarks within sensor range
    std::vector<LandmarkObs> predictions;
    for (const auto &lm : map_landmarks.landmark_list) {
      double d = dist(p.x, p.y, lm.x_f, lm.y_f);
      if (d <= sensor_range) {
        LandmarkObs pred;
        pred.id = lm.id_i;
        pred.x = lm.x_f;
        pred.y = lm.y_f;
        predictions.push_back(pred);
      }
    }

    if (predictions.empty()) {
      p.weight = 1e-8;
      weights[i] = p.weight;
      continue;
    }

    // 2) Transform observations from vehicle to map coordinates
    std::vector<LandmarkObs> trans_observations;
    for (const auto &obs : observations) {
      LandmarkObs t_obs;
      t_obs.id = obs.id;
      t_obs.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
      t_obs.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
      trans_observations.push_back(t_obs);
    }

    // 3) Associate transformed observations with landmarks
    dataAssociation(predictions, trans_observations);

    // 4) Compute weight using multivariate Gaussian
    double weight = 1.0;

    for (const auto &t_obs : trans_observations) {
      // find associated prediction landmark
      auto it = std::find_if(predictions.begin(), predictions.end(),
                             [&t_obs](const LandmarkObs &pred) {
                               return pred.id == t_obs.id;
                             });
      if (it == predictions.end()) {
        // no associated landmark found, skip or assign tiny prob
        weight *= 1e-8;
        continue;
      }

      double mu_x = it->x;
      double mu_y = it->y;
      double dx = t_obs.x - mu_x;
      double dy = t_obs.y - mu_y;

      double exponent = (dx * dx) / (2.0 * sig_x2) +
                        (dy * dy) / (2.0 * sig_y2);

      double obs_w = gauss_norm * std::exp(-exponent);
      if (obs_w < 1e-12) {
        obs_w = 1e-12;
      }

      weight *= obs_w;
    }

    p.weight = weight;
    weights[i] = weight;
  }

  // Optional: normalize weights (not strictly required for std::discrete_distribution)
  double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (weight_sum > 0.0) {
    for (int i = 0; i < num_particles; ++i) {
      particles[i].weight /= weight_sum;
      weights[i] /= weight_sum;
    }
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles;
  new_particles.reserve(num_particles);

  std::discrete_distribution<int> dist_index(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    int idx = dist_index(gen);
    new_particles.push_back(particles[idx]);
  }

  particles = std::move(new_particles);
}
