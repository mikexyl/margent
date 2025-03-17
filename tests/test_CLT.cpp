#include <aria_common/logging.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/BayesTree.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/dataset.h>

#include <iostream>
#include <limits>

#include "margent/MarginalsExposeBayesTree.h"

double computeMutualInformationFromInformationMatrix(
    const gtsam::Matrix& infoMatrix,
    size_t i,
    size_t j,
    size_t poseDim) {
  // Extract relevant submatrices
  gtsam::Matrix H_ii =
      infoMatrix.block(i * poseDim, i * poseDim, poseDim, poseDim);
  gtsam::Matrix H_jj =
      infoMatrix.block(j * poseDim, j * poseDim, poseDim, poseDim);
  gtsam::Matrix H_ij =
      infoMatrix.block(i * poseDim, j * poseDim, poseDim, poseDim);

  // Compute determinants (Hessian values, not covariance)
  double det_Hii = H_ii.determinant();
  double det_Hjj = H_jj.determinant();
  double det_Hij =
      (H_ii * H_jj - H_ij * H_ij).determinant();  // Schur complement

  // Avoid numerical instability
  if (det_Hii <= 0 || det_Hjj <= 0 || det_Hij <= 0) {
    return 0.0;
  }

  // Compute mutual information using the Hessian instead of covariance
  return 0.5 * std::log(det_Hii * det_Hjj / det_Hij);
}

// Function to compute mutual information from covariance matrix for poses
double computeMutualInformation(const gtsam::Matrix& covariance,
                                size_t i,
                                size_t j,
                                size_t poseDim) {
  gtsam::Matrix subCov =
      covariance.block(i * poseDim, j * poseDim, poseDim, poseDim);
  gtsam::Vector variances = subCov.diagonal();

  // Ensure variances are positive to prevent NaNs
  if ((variances.array() <= 0.0).any()) {
    return 0.0;
  }

  // Compute determinant-based mutual information
  double det_ii = subCov.block(0, 0, poseDim, poseDim).determinant();
  double det_jj = covariance.block(j * poseDim, j * poseDim, poseDim, poseDim)
                      .determinant();
  double det_ij = subCov.determinant();

  // Avoid numerical issues
  if (det_ii <= 0 || det_jj <= 0 || det_ij <= 0) {
    return 0.0;
  }

  return 0.5 * std::log(det_ii * det_jj / det_ij);
}

// Function to build the Chow-Liu Tree using Prim's algorithm
std::vector<std::pair<size_t, size_t>>
buildChowLiuTree(const gtsam::Matrix& info, size_t numPoses, size_t poseDim) {
  // Compute Mutual Information (MI) for all pose pairs
  std::vector<std::vector<double>> MI(numPoses,
                                      std::vector<double>(numPoses, 0.0));
  for (size_t i = 0; i < numPoses; ++i) {
    for (size_t j = i + 1; j < numPoses; ++j) {
      MI[i][j] = MI[j][i] =
          computeMutualInformationFromInformationMatrix(info, i, j, poseDim);
    }
  }

  // Prim's Algorithm to find Maximum Spanning Tree (MST)
  std::vector<std::pair<size_t, size_t>> treeEdges;
  std::vector<bool> inTree(numPoses, false);
  std::vector<double> maxWeight(numPoses,
                                -std::numeric_limits<double>::infinity());
  std::vector<size_t> parent(numPoses, -1);

  // Start from node 0
  maxWeight[0] = 0;
  inTree[0] = true;

  // Initialize maxWeight values with direct MI connections
  for (size_t v = 0; v < numPoses; ++v) {
    maxWeight[v] = MI[0][v];
    parent[v] = 0;
  }

  for (size_t count = 0; count < numPoses - 1; ++count) {
    // Find node with max weight that is not yet in the tree
    size_t maxNode = -1;
    double maxValue = -std::numeric_limits<double>::infinity();
    for (size_t v = 0; v < numPoses; ++v) {
      if (!inTree[v] && maxWeight[v] > maxValue) {
        maxValue = maxWeight[v];
        maxNode = v;
      }
    }

    // Mark as in the tree
    if (maxNode == static_cast<size_t>(-1)) {
      break;  // Prevent accessing invalid node
    }
    inTree[maxNode] = true;
    if (parent[maxNode] != static_cast<size_t>(-1)) {
      treeEdges.emplace_back(parent[maxNode], maxNode);
    }

    // Update adjacent nodes
    for (size_t v = 0; v < numPoses; ++v) {
      if (!inTree[v] && MI[maxNode][v] > maxWeight[v]) {
        maxWeight[v] = MI[maxNode][v];
        parent[v] = maxNode;
      }
    }
  }

  return treeEdges;
}

// Main function to process a clique and compute Chow-Liu tree probabilities
void processClique(const gtsam::GaussianBayesTree::Clique& clique,
                   size_t poseDim) {
  // Extract conditional
  auto conditional = clique.conditional();

  // Determine the number of poses
  size_t numPoses = conditional->keys().size();

  // Build Chow-Liu Tree for pose-wise structure
  std::vector<std::pair<size_t, size_t>> chowLiuTree =
      buildChowLiuTree(conditional->information(), numPoses, poseDim);

  std::cout << fmt::format("Clique: \n  {}\n", clique);

  // Print Chow-Liu Tree edges
  std::cout << "Chow-Liu Tree edges (Pose-wise):\n";
  auto keys = conditional->keys();
  for (const auto& edge : chowLiuTree) {
    std::cout << keys[edge.first] << " -- " << keys[edge.second] << "\n";
  }
}

int main() {
  auto [graph, values] = readG2o("smallGrid3D.g2o", true);
  // add prior to first pose
  graph->add(PriorFactor<Pose3>(0, Pose3(), noiseModel::Unit::Create(6)));
  // optimize
  GaussNewtonOptimizer optimizer(*graph, *values);
  *values = optimizer.optimize();

  MarginalsExposeBayesTree marginals(*graph, *values);
  auto bayesTree = marginals.getBayesTree();

  std::set<GaussianBayesTreeClique::shared_ptr> cliques;
  for (const auto& [_, clique] : bayesTree.nodes()) {
    if (!clique) continue;
    if (cliques.contains(clique)) continue;
    processClique(*clique, 6);
    cliques.insert(clique);
  }
}