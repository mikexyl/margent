#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/dataset.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace gtsam;

class GBPMarginals {
 private:
  NonlinearFactorGraph graph_;
  Values linearizationPoint_;
  // Current marginal covariance for each variable.
  std::unordered_map<Key, Matrix> marginals_;

  size_t maxIterations_ = 100;
  double convergenceThreshold_ = 1e-4;
  double dampingFactor_ = 0.9;

 public:
  GBPMarginals(const NonlinearFactorGraph& graph,
               const Values& linearizationPoint)
      : graph_(graph), linearizationPoint_(linearizationPoint) {
    // Initialize each marginal with a large covariance (i.e. low information)
    for (const auto& key_value : linearizationPoint_) {
      size_t dim = linearizationPoint_.at(key_value.key).dim();
      marginals_[key_value.key] = Matrix::Identity(dim, dim) * 1e6;
    }
    computeMarginalsGBP();
  }

  // Iteratively update marginal estimates using GBP-style updates with Schur
  // complement.
  void computeMarginalsGBP() {
    auto linearGraph = graph_.linearize(linearizationPoint_);
    for (size_t iter = 0; iter < maxIterations_; ++iter) {
      bool converged = true;
      std::unordered_map<Key, Matrix> newMarginals = marginals_;
      // Loop over all factors in the linearized graph.
      for (int i = 0; i < graph_.size(); ++i) {
        auto factor = graph_.at(i);
        auto noise_factor =
            boost::dynamic_pointer_cast<NoiseModelFactor>(factor);
        auto noiseModel = noise_factor->noiseModel();
        auto gaussian =
            boost::dynamic_pointer_cast<noiseModel::Gaussian>(noiseModel);
        // Process only JacobianFactors.
        boost::shared_ptr<JacobianFactor> jf =
            boost::dynamic_pointer_cast<JacobianFactor>(linearGraph->at(i));
        if (!jf) {
          throw std::runtime_error("Factor is not a JacobianFactor");
        }
        KeyVector keys = jf->keys();
        // For each variable in this factor, compute its updated marginal via
        // Schur complement.
        for (Key key : keys) {
          Matrix updatedMarginal =
              computeSchurMarginalForKey(jf, *gaussian, key);
          Matrix dampedMarginal = dampingFactor_ * updatedMarginal +
                                  (1 - dampingFactor_) * marginals_[key];
          if ((dampedMarginal - marginals_[key]).norm() >
              convergenceThreshold_) {
            converged = false;
          }
          newMarginals[key] = dampedMarginal;
        }
      }
      marginals_ = newMarginals;
      if (converged) {
        std::cout << "GBP converged at iteration " << iter << std::endl;
        break;
      }
    }
  }

  // For a given factor and a target variable, compute its updated marginal
  // covariance using the Schur complement.
  Matrix computeSchurMarginalForKey(const boost::shared_ptr<JacobianFactor>& jf,
                                    const noiseModel::Gaussian& noiseModel,
                                    Key target) {
    // if target is the only key in the factor, return the noise model
    if (jf->keys().size() == 1) {
      return noiseModel.information().inverse();
    }

    // Get all keys (neighbors) in this factor.
    KeyVector neighborKeys = jf->keys();
    std::unordered_map<Key, int> offsets;
    std::unordered_map<Key, int> dims;
    int totalDim = 0;
    for (Key k : neighborKeys) {
      int d = linearizationPoint_.at(k).dim();
      dims[k] = d;
      offsets[k] = totalDim;
      totalDim += d;
    }
    // Build the joint information matrix from the current marginal estimates.
    // Each block is the inverse of the current covariance.
    Matrix jointInformation = Matrix::Zero(totalDim, totalDim);
    for (Key k : neighborKeys) {
      int off = offsets[k];
      int d = dims[k];
      if (k == target) {
        jointInformation.block(off, off, d, d) = Matrix::Identity(d, d) * 1e-2;
      } else {
        Matrix info = marginals_.at(k).inverse();
        jointInformation.block(off, off, d, d) = info;
      }
    }
    // Get the factor's contribution.
    Matrix A = jf->getA();
    Matrix factorInformation;
    factorInformation = A.transpose() * noiseModel.information() * A;

    // Combine to form the updated joint information.
    Matrix updatedJointInformation = jointInformation + factorInformation;
    // set target variable block to zero
    // For the target variable, extract its block using Schur complement.
    int t_start = offsets[target];
    int t_dim = dims[target];
    Matrix updatedMarginal =
        computeSchurComplement(updatedJointInformation, t_start, t_dim);
    updatedMarginal = TransformCovariance<Pose3>(
        linearizationPoint_.at<Pose3>(target))(updatedMarginal);
    return updatedMarginal;
  }

  // Given a joint information matrix I of size totalDim x totalDim,
  // where the target variable occupies the block starting at t_start of size
  // t_dim, compute the Schur complement and return the marginal covariance for
  // the target variable.
  Matrix computeSchurComplement(const Matrix& I, int t_start, int t_dim) {
    int totalDim = I.rows();
    // Build a vector of indices corresponding to all variables other than the
    // target.
    std::vector<int> otherIndices;
    for (int i = 0; i < totalDim; ++i) {
      if (i < t_start || i >= t_start + t_dim) {
        otherIndices.push_back(i);
      }
    }
    // I_tt: block for target variable.
    Matrix I_tt = I.block(t_start, t_start, t_dim, t_dim);

    int otherSize = static_cast<int>(otherIndices.size());
    Matrix I_tR(t_dim, otherSize);
    Matrix I_Rt(otherSize, t_dim);
    Matrix I_RR(otherSize, otherSize);
    for (int i = 0; i < otherSize; ++i) {
      for (int j = 0; j < otherSize; ++j) {
        I_RR(i, j) = I(otherIndices[i], otherIndices[j]);
      }
      for (int j = 0; j < t_dim; ++j) {
        I_tR(j, i) = I(t_start + j, otherIndices[i]);
        I_Rt(i, j) = I(otherIndices[i], t_start + j);
      }
    }
    // Compute the Schur complement: I_s = I_tt - I_tR * I_RR^{-1} * I_Rt.
    Matrix I_RR_inv = I_RR.inverse();
    Matrix I_s = I_tt - I_tR * I_RR_inv * I_Rt;
    // The updated marginal covariance for the target is the inverse of I_s.
    Matrix marginal = I_s.inverse();

    return marginal;
  }

  Matrix marginalCovariance(Key key) const {
    auto it = marginals_.find(key);
    if (it == marginals_.end()) {
      throw std::runtime_error("Key not found in marginals");
    }
    return it->second;
  }
};

// Function to compare two sets of marginal covariances using the Frobenius
// norm.
void compareCovariances(
    const std::map<gtsam::Key, gtsam::Matrix>& marginalsGTSAM,
    const std::map<gtsam::Key, gtsam::Matrix>& marginalsGBP) {
  double totalDifference = 0.0;
  int count = 0;

  for (const auto& kv : marginalsGTSAM) {
    gtsam::Key key = kv.first;
    const gtsam::Matrix& covGTSAM = kv.second;

    // Check if the key exists in the GBP marginals.
    auto it = marginalsGBP.find(key);
    if (it != marginalsGBP.end()) {
      const gtsam::Matrix& covGBP = it->second;

      // Compute the difference and then its Frobenius norm.
      double diffNorm = (covGBP).trace() / covGTSAM.trace();
      totalDifference += diffNorm;
      count++;
    } else {
      std::cout << "Key " << key << " not found in GBP marginals." << std::endl;
    }
  }

  if (count > 0) {
    std::cout << "Average Frobenius norm difference: "
              << totalDifference / count << std::endl;
  } else {
    std::cout << "No common keys to compare." << std::endl;
  }
}

void saveMarginalsAsG2o(std::string file, std::map<Key, Matrix> marginals) {
  NonlinearFactorGraph graph;
  Values values;

  for (const auto& kv : marginals) {
    Key key = kv.first;
    Matrix cov = kv.second;
    graph.add(BetweenFactor<Pose3>(
        key, key, Pose3(), noiseModel::Gaussian::Covariance(cov)));
    values.insert(key, Pose3());
  }

  writeG2o(graph, values, file);
}

namespace po = boost::program_options;

// Usage example:
int main(int argc, char** argv) {
  std::string input_file;

  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "input-file,i",
      po::value<std::string>(&input_file),
      "input file containing the factor graph");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Read the graph and initial estimate from a G2O file.
  auto [graph, initialEstimate] = readG2o(input_file, true);
  // Add a prior factor to fix the first pose.
  graph->add(PriorFactor<Pose3>(0,
                                initialEstimate->at<Pose3>(0),
                                noiseModel::Diagonal::Sigmas(Vector6(
                                    1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))));
  // Optimize the nonlinear factor graph.
  LevenbergMarquardtOptimizer optimizer(*graph, *initialEstimate);
  Values result = optimizer.optimize();

  // Compute marginal covariances using our GBP-based method.
  auto startGBP = std::chrono::high_resolution_clock::now();
  GBPMarginals gbpMarginals(*graph, result);
  std::map<Key, Matrix> marginalsGBP;
  // compute covariances for all keys
  for (const auto& key : result.keys()) {
    marginalsGBP[key] = gbpMarginals.marginalCovariance(key);
  }
  auto endGBP = std::chrono::high_resolution_clock::now();
  auto durationGBP =
      std::chrono::duration_cast<std::chrono::microseconds>(endGBP - startGBP)
          .count();

  // Also compute built-in marginal covariances.
  auto startGTSAM = std::chrono::high_resolution_clock::now();
  Marginals builtInMarginals(*graph, result);
  // compute covariances for all keys
  std::map<Key, Matrix> marginalsGTSAM;
  for (const auto& key : result.keys()) {
    marginalsGTSAM[key] = builtInMarginals.marginalCovariance(key);
  }
  auto endGTSAM = std::chrono::high_resolution_clock::now();
  auto durationGTSAM = std::chrono::duration_cast<std::chrono::microseconds>(
                           endGTSAM - startGTSAM)
                           .count();

  std::cout << "Built-in Marginal Covariance spent " << durationGTSAM << " us"
            << std::endl;
  std::cout << "GBP Marginal Covariance spent " << durationGBP << " us"
            << std::endl;

  compareCovariances(marginalsGTSAM, marginalsGBP);

  // print one of the marginals
  Key random_key = result.keys().at(rand() % result.size());
  std::cout << "Marginal covariance for key " << random_key << " using GBP: \n"
            << marginalsGBP[random_key] << std::endl;
  std::cout << "Marginal covariance for key " << random_key
            << " using built-in: \n"
            << marginalsGTSAM[random_key] << std::endl;

  saveMarginalsAsG2o(input_file + ".marginals", marginalsGBP);

  return 0;
}
