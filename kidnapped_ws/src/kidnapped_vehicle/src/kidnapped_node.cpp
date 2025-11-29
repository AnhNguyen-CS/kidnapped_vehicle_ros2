#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <chrono>
#include <random>
#include <cmath>

#include "kidnapped_vehicle/particle_filter.h"
#include "kidnapped_vehicle/helper_functions.h"

using namespace std::chrono_literals;

class KidnappedNode : public rclcpp::Node {
public:
  KidnappedNode()
  : Node("kidnapped_node"),
    pf_(),
    pf_initialized_(false),
    step_(0),
    max_steps_(100),
    delta_t_(0.1),
    velocity_(0.6),
    yaw_rate_(0.05),
    sensor_range_(40.0),
    max_translation_error_(0.5),
    max_yaw_error_(0.05),
    max_trans_err_(0.0),
    max_yaw_err_(0.0),
    true_x_(102.0),
    true_y_(65.0),
    true_theta_(0.0)
  {
    RCLCPP_INFO(this->get_logger(), "Kidnapped vehicle node started.");

    // Standard deviations for initial GPS and motion model
    std_pos[0] = 0.01;   // [m]
    std_pos[1] = 0.01;   // [m]
    std_pos[2] = 0.001;  // [rad]

    // Standard deviations for landmark observations
    std_landmark[0] = 0.01; // [m]
    std_landmark[1] = 0.01; // [m]

    // Load map data
    std::string map_file = "/root/kidnapped_ws/src/kidnapped_vehicle/data/map_data.txt";
    if (!read_map_data(map_file, map_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load map file: %s", map_file.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "Loaded %zu landmarks from %s",
                  map_.landmark_list.size(), map_file.c_str());
    }

    // Publishers for RViz visualization
    particles_pub_   = this->create_publisher<visualization_msgs::msg::Marker>("particles", 1);
    traj_pub_        = this->create_publisher<visualization_msgs::msg::Marker>("trajectory", 1);
    pose_pub_        = this->create_publisher<geometry_msgs::msg::PoseStamped>("pf_pose", 1);
    landmarks_pub_   = this->create_publisher<visualization_msgs::msg::Marker>("landmarks", 1);
    gt_traj_pub_     = this->create_publisher<visualization_msgs::msg::Marker>("ground_truth", 1);

    // Initialize trajectory marker (PF estimate)
    traj_marker_.header.frame_id = "map";
    traj_marker_.ns = "trajectory";
    traj_marker_.id = 0;
    traj_marker_.type = visualization_msgs::msg::Marker::LINE_STRIP;
    traj_marker_.action = visualization_msgs::msg::Marker::ADD;
    traj_marker_.scale.x = 0.3;  // line width
    traj_marker_.color.a = 1.0;
    traj_marker_.color.r = 1.0;
    traj_marker_.color.g = 0.0;
    traj_marker_.color.b = 0.0;

    // Initialize ground-truth trajectory marker
    gt_traj_marker_.header.frame_id = "map";
    gt_traj_marker_.ns = "ground_truth";
    gt_traj_marker_.id = 0;
    gt_traj_marker_.type = visualization_msgs::msg::Marker::LINE_STRIP;
    gt_traj_marker_.action = visualization_msgs::msg::Marker::ADD;
    gt_traj_marker_.scale.x = 0.3;  // line width
    gt_traj_marker_.color.a = 1.0;
    gt_traj_marker_.color.r = 1.0;  // red
    gt_traj_marker_.color.g = 0.0;
    gt_traj_marker_.color.b = 0.0;

    // Pre-build landmarks marker (blue spheres)
    landmarks_marker_.header.frame_id = "map";
    landmarks_marker_.ns = "landmarks";
    landmarks_marker_.id = 0;
    landmarks_marker_.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    landmarks_marker_.action = visualization_msgs::msg::Marker::ADD;
    landmarks_marker_.scale.x = 0.8;
    landmarks_marker_.scale.y = 0.8;
    landmarks_marker_.scale.z = 0.8;
    landmarks_marker_.color.a = 1.0;
    landmarks_marker_.color.r = 0.0;
    landmarks_marker_.color.g = 0.0;
    landmarks_marker_.color.b = 1.0;  // blue

    for (const auto &lm : map_.landmark_list) {
      geometry_msgs::msg::Point p;
      p.x = lm.x_f;
      p.y = lm.y_f;
      p.z = 0.0;
      landmarks_marker_.points.push_back(p);
    }

    // Initialize particle filter with a noisy GPS estimate around true pose
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

    double init_x = true_x_ + dist_x(gen_);
    double init_y = true_y_ + dist_y(gen_);
    double init_theta = true_theta_ + dist_theta(gen_);

    pf_.init(init_x, init_y, init_theta, std_pos);
    pf_initialized_ = true;
    start_time_ = this->now();

    RCLCPP_INFO(this->get_logger(),
                "PF initialized at x=%.2f y=%.2f theta=%.2f (true x=%.2f y=%.2f theta=%.2f)",
                init_x, init_y, init_theta, true_x_, true_y_, true_theta_);

    // Timer to run filter at fixed delta_t_
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(delta_t_),
      std::bind(&KidnappedNode::timerCallback, this));
  }

private:
  void timerCallback()
  {
    if (!pf_initialized_) {
      return;
    }

    // 1. Move the "true" vehicle (ground truth, no noise)
    moveTrueVehicle();

    // 2. Simulate sensor observations from true pose
    std::vector<LandmarkObs> observations = simulateObservations();

    // 3. Run particle filter: prediction, update, resample
    pf_.prediction(delta_t_, std_pos, velocity_, yaw_rate_);
    pf_.updateWeights(sensor_range_, std_landmark, observations, map_);
    pf_.resample();

    // 4. Get best particle (estimate)
    Particle best = getBestParticle();

    // 5. Track error vs ground truth
    updateErrors(best);

    // 6. Publish visualization markers
    publishParticles();
    publishPose(best);
    publishTrajectory(best);
    publishLandmarks();              // landmarks (static)
    publishGroundTruthTrajectory();  // ground truth line

    // 7. Check stopping condition
    step_++;
    if (step_ >= max_steps_) {
      double elapsed = (this->now() - start_time_).seconds();
      RCLCPP_INFO(this->get_logger(),
                  "Finished simulation. Steps=%d, time=%.2f s", step_, elapsed);
      RCLCPP_INFO(this->get_logger(),
                  "Max translation error = %.3f (limit %.3f)",
                  max_trans_err_, max_translation_error_);
      RCLCPP_INFO(this->get_logger(),
                  "Max yaw error         = %.3f (limit %.3f)",
                  max_yaw_err_, max_yaw_error_);

      if (max_trans_err_ < max_translation_error_ && max_yaw_err_ < max_yaw_error_) {
        RCLCPP_INFO(this->get_logger(), "PASS: PF meets accuracy requirements.");
      } else {
        RCLCPP_WARN(this->get_logger(), "FAIL: PF does NOT meet accuracy requirements.");
      }

      rclcpp::shutdown();
    }
  }

  void moveTrueVehicle()
  {
    // Bicycle model without noise
    if (std::fabs(yaw_rate_) > 1e-5) {
      true_x_ += (velocity_ / yaw_rate_) *
                 (std::sin(true_theta_ + yaw_rate_ * delta_t_) - std::sin(true_theta_));
      true_y_ += (velocity_ / yaw_rate_) *
                 (-std::cos(true_theta_ + yaw_rate_ * delta_t_) + std::cos(true_theta_));
      true_theta_ += yaw_rate_ * delta_t_;
    } else {
      true_x_ += velocity_ * delta_t_ * std::cos(true_theta_);
      true_y_ += velocity_ * delta_t_ * std::sin(true_theta_);
    }
    true_theta_ = normalize_angle(true_theta_);
  }

  std::vector<LandmarkObs> simulateObservations()
  {
    std::vector<LandmarkObs> observations;

    std::normal_distribution<double> dist_x(0.0, std_landmark[0]);
    std::normal_distribution<double> dist_y(0.0, std_landmark[1]);

    for (const auto &lm : map_.landmark_list) {
      double dx = lm.x_f - true_x_;
      double dy = lm.y_f - true_y_;
      double range = std::sqrt(dx * dx + dy * dy);

      if (range <= sensor_range_) {
        // Transform landmark to vehicle coordinates
        double x_rel =  std::cos(-true_theta_) * dx - std::sin(-true_theta_) * dy;
        double y_rel =  std::sin(-true_theta_) * dx + std::cos(-true_theta_) * dy;

        LandmarkObs obs;
        obs.id = lm.id_i;
        obs.x = x_rel + dist_x(gen_);
        obs.y = y_rel + dist_y(gen_);

        observations.push_back(obs);
      }
    }

    return observations;
  }

  Particle getBestParticle()
  {
    double max_w = -1.0;
    Particle best;
    for (const auto &p : pf_.particles) {
      if (p.weight > max_w) {
        max_w = p.weight;
        best = p;
      }
    }
    return best;
  }

  void updateErrors(const Particle &best)
  {
    double dx = best.x - true_x_;
    double dy = best.y - true_y_;
    double trans_err = std::sqrt(dx * dx + dy * dy);
    double yaw_err = std::fabs(normalize_angle(best.theta - true_theta_));

    if (trans_err > max_trans_err_) {
      max_trans_err_ = trans_err;
    }
    if (yaw_err > max_yaw_err_) {
      max_yaw_err_ = yaw_err;
    }

    RCLCPP_INFO(this->get_logger(),
                "step=%d true(%.2f, %.2f, %.2f) est(%.2f, %.2f, %.2f) "
                "trans_err=%.3f yaw_err=%.3f",
                step_,
                true_x_, true_y_, true_theta_,
                best.x, best.y, best.theta,
                trans_err, yaw_err);
  }

  void publishParticles()
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = this->now();
    m.ns = "particles";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::POINTS;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.3;
    m.scale.y = 0.3;
    m.color.a = 1.0;
    m.color.g = 1.0;  // green

    m.points.clear();
    for (const auto &p : pf_.particles) {
      geometry_msgs::msg::Point pt;
      pt.x = p.x;
      pt.y = p.y;
      pt.z = 0.0;
      m.points.push_back(pt);
    }

    particles_pub_->publish(m);
  }

  void publishPose(const Particle &best)
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.header.stamp = this->now();
    pose.pose.position.x = best.x;
    pose.pose.position.y = best.y;
    pose.pose.position.z = 0.0;

    double yaw = best.theta;
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = std::sin(yaw / 2.0);
    pose.pose.orientation.w = std::cos(yaw / 2.0);

    pose_pub_->publish(pose);
  }

  void publishTrajectory(const Particle &best)
  {
    traj_marker_.header.stamp = this->now();

    geometry_msgs::msg::Point pt;
    pt.x = best.x;
    pt.y = best.y;
    pt.z = 0.0;
    traj_marker_.points.push_back(pt);

    traj_pub_->publish(traj_marker_);
  }

  void publishLandmarks()
  {
    // Just stamp and publish the pre-built marker
    landmarks_marker_.header.stamp = this->now();
    landmarks_pub_->publish(landmarks_marker_);
  }

  void publishGroundTruthTrajectory()
  {
    gt_traj_marker_.header.stamp = this->now();

    geometry_msgs::msg::Point pt;
    pt.x = true_x_;
    pt.y = true_y_;
    pt.z = 0.0;
    gt_traj_marker_.points.push_back(pt);

    gt_traj_pub_->publish(gt_traj_marker_);
  }

  // Members
  ParticleFilter pf_;
  bool pf_initialized_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr particles_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr landmarks_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr gt_traj_pub_;

  // Markers for RViz
  visualization_msgs::msg::Marker traj_marker_;      // PF trajectory (estimated)
  visualization_msgs::msg::Marker gt_traj_marker_;   // Ground truth trajectory
  visualization_msgs::msg::Marker landmarks_marker_; // Landmarks

  // Random generator
  std::mt19937 gen_{std::random_device{}()};

  // Noise parameters
  double std_pos[3]      = {0.1, 0.1, 0.005};
  double std_landmark[2] = {0.05, 0.05};

  // Simulation
  int step_;
  int max_steps_;
  double delta_t_;
  double velocity_;
  double yaw_rate_;
  double sensor_range_;

  // Ground truth
  double true_x_;
  double true_y_;
  double true_theta_;

  // Assignment thresholds
  double max_translation_error_;
  double max_yaw_error_;

  // Tracked maximum errors
  double max_trans_err_;
  double max_yaw_err_;

  // Map with landmarks
  Map map_;

  // Timer for performance measurement
  rclcpp::Time start_time_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<KidnappedNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
