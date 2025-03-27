#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <aria_common/logging.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/dataset.h>

#include <iostream>

#include "margent/MarginalsExposeBayesTree.h"

using namespace gtsam;

/**
 * @brief save marginals as "between factor to itself" in g2o format
 *
 * @param values
 * @param covs
 * @param output_path
 */
inline void saveMarginalsG2o(const Values& results,
                             const std::map<Key, Matrix>& marginal_covariances,
                             const std::string& output_path) {
  NonlinearFactorGraph marginal_factors;
  for (auto const& key : results.keySet()) {
    if (not marginal_covariances.contains(key)) {
      continue;
    }
    auto noise = noiseModel::Gaussian::Covariance(marginal_covariances.at(key));
    // create between factor to save in g2o
    marginal_factors.add(BetweenFactor<Pose3>(key, key, Pose3(), noise));
  }

  writeG2o(marginal_factors, results, output_path);
}

// Define command-line flags for filenames
ABSL_FLAG(std::string, g2o_file, "sphere2500.g2o", "Path to the g2o file");
ABSL_FLAG(std::string,
          gt_marginals_file,
          "sphere2500.g2o.marginals",
          "Path to the ground truth marginals file");

int main(int argc, char** argv) {
  // Parse command-line flags
  absl::ParseCommandLine(argc, argv);

  // Retrieve flag values
  std::string g2o_file = absl::GetFlag(FLAGS_g2o_file);
  std::string gt_marginals_file = absl::GetFlag(FLAGS_gt_marginals_file);

  // Load the graph
  auto [graph, initial] = readG2o(g2o_file, true);

  // compute the marginals
  // add prior at 0
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
  graph->add(PriorFactor<Pose3>(0, initial->at<Pose3>(0), prior_noise));

  auto optimizer = GaussNewtonOptimizer(*graph, *initial);
  *initial = optimizer.optimize();

  auto marginals_approx =
      MarginalsExposeBayesTree::FromOptimalValues(*graph, *initial);

  Values perturbed{};
  for (auto key : initial->keySet()) {
    auto value = initial->at<Pose3>(key);
    auto perturbed_value =
        Pose3(value.rotation(), value.translation() + Vector3::Random());
    perturbed.insert(key, perturbed_value);
  }

  auto time_started = std::chrono::high_resolution_clock::now();
  auto marginals_perturbed =
      marginals_approx.marginalCovariance(perturbed.keySet(), perturbed);
  auto time_finished = std::chrono::high_resolution_clock::now();
  std::cout << "Time to compute marginals with approximation: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_finished - time_started)
                   .count()
            << "ms" << std::endl;

  time_started = std::chrono::high_resolution_clock::now();
  Marginals marginals(*graph, perturbed);
  auto covs = marginals.marginalCovariance(initial->keySet());
  time_finished = std::chrono::high_resolution_clock::now();
  std::cout << "Time to compute marginals without approximation: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   time_finished - time_started)
                   .count()
            << "ms" << std::endl;

  double frobenius_norm_error = 0;
  std::vector<double> frobenius_norm_errors;
  for (auto const& [key, cov] : marginals_perturbed) {
    CHECK(covs.contains(key));
    auto full_cov = covs.at(key);
    auto diff = cov - full_cov;
    frobenius_norm_errors.push_back(diff.norm() / full_cov.norm());
  }

  // get statistics of frobenius norm errors
  double sum = std::accumulate(
      frobenius_norm_errors.begin(), frobenius_norm_errors.end(), 0.0);
  double mean = sum / frobenius_norm_errors.size();
  double max = *std::max_element(frobenius_norm_errors.begin(),
                                 frobenius_norm_errors.end());
  double min = *std::min_element(frobenius_norm_errors.begin(),
                                 frobenius_norm_errors.end());
  double std = 0;
  for (auto const& error : frobenius_norm_errors) {
    std += std::pow(error - mean, 2);
  }
  std = std::sqrt(std / frobenius_norm_errors.size());
  std::cout << "Frobenius norm error statistics: mean: " << mean
            << ", max: " << max << ", min: " << min << ", std: " << std
            << std::endl;

  return 0;
}
