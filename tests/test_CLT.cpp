#include <aria_common/logging.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/BayesTree.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <iostream>
#include <limits>

#include "margent/MarginalsExposeBayesTree.h"

using namespace gtsam;

double computeMutualInformationFromInformationMatrix(const Matrix& infoMatrix,
                                                     size_t i,
                                                     size_t j,
                                                     size_t poseDim) {
  // Extract submatrices from the information matrix
  Matrix H_ii = infoMatrix.block(i * poseDim, i * poseDim, poseDim, poseDim);
  Matrix H_jj = infoMatrix.block(j * poseDim, j * poseDim, poseDim, poseDim);
  Matrix H_ij = infoMatrix.block(i * poseDim, j * poseDim, poseDim, poseDim);

  // Compute determinants
  double det_Hii = H_ii.determinant();
  double det_Hjj = H_jj.determinant();
  double det_Hij =
      (H_ii * H_jj - H_ij * H_ij).determinant();  // Schur complement

  // Ensure determinants are positive and prevent log(0) or log(negative)
  if (det_Hii <= 0 || det_Hjj <= 0 || det_Hij <= 0) {
    return 0.0;  // Avoid invalid log computation
  }

  // Compute determinant ratio (FLIPPED from covariance formula)
  double MI = det_Hij / (det_Hii * det_Hjj);
  CHECK(MI >= 0,
        fmt::format("MI should be non-negative, {}\n, det_Hii = {}, det_Hjj = "
                    "{}, det_Hij = {}",
                    infoMatrix,
                    det_Hii,
                    det_Hjj,
                    det_Hij));

  // Compute mutual information using the Hessian instead of covariance
  return MI;
}

// Function to compute mutual information from covariance matrix for poses
double computeMutualInformation(const Matrix& covariance,
                                size_t i,
                                size_t j,
                                size_t poseDim) {
  Matrix subCov = covariance.block(i * poseDim, j * poseDim, poseDim, poseDim);
  Vector variances = subCov.diagonal();

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
struct ChowLiuTreeResult {
  std::vector<std::pair<size_t, size_t>> treeEdges;
  double totalMutualInformation;      // MI sum from CLT
  double fullGraphMutualInformation;  // MI sum before CLT
};

ChowLiuTreeResult buildChowLiuTree(const Matrix& infoMatrix,
                                   size_t numPoses,
                                   size_t poseDim) {
  // Compute Mutual Information (MI) for all pose pairs
  std::vector<std::vector<double>> MI(numPoses,
                                      std::vector<double>(numPoses, 0.0));
  double fullGraphMutualInformation = 0.0;  // Store total MI before CLT

  for (size_t i = 0; i < numPoses; ++i) {
    for (size_t j = i + 1; j < numPoses; ++j) {
      MI[i][j] = MI[j][i] = computeMutualInformationFromInformationMatrix(
          infoMatrix, i, j, poseDim);
      fullGraphMutualInformation += MI[i][j];  // Sum all MI values before CLT
    }
  }

  // Prim's Algorithm to find Maximum-Weight Spanning Tree (MWST)
  std::vector<std::pair<size_t, size_t>> treeEdges;
  std::vector<bool> inTree(numPoses, false);
  std::vector<double> maxWeight(numPoses,
                                -std::numeric_limits<double>::infinity());
  std::vector<size_t> parent(numPoses, -1);
  double totalMutualInformation = 0.0;  // Store MI sum from CLT

  // Start from node 0
  maxWeight[0] = 0;
  inTree[0] = true;

  // Initialize maxWeight values with direct MI connections
  for (size_t v = 0; v < numPoses; ++v) {
    maxWeight[v] = MI[0][v];
    parent[v] = 0;
  }

  for (size_t count = 0; count < numPoses - 1; ++count) {
    // Find node with max MI that is not yet in the tree
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
      totalMutualInformation += maxWeight[maxNode];  // Sum MI of selected edges
    }

    // Update adjacent nodes
    for (size_t v = 0; v < numPoses; ++v) {
      if (!inTree[v] && MI[maxNode][v] > maxWeight[v]) {
        maxWeight[v] = MI[maxNode][v];
        parent[v] = maxNode;
      }
    }
  }

  static constexpr double kMinLostMI = 0.0;
  if ((totalMutualInformation / fullGraphMutualInformation) < kMinLostMI) {
    std::cout << fmt::format("Restoring lost edges, MI: {:.3f}/{:.3f}",
                             totalMutualInformation,
                             fullGraphMutualInformation)
              << std::endl;
    std::vector<std::pair<size_t, size_t>> lostEdges;
    for (size_t i = 0; i < numPoses; ++i) {
      for (size_t j = i + 1; j < numPoses; ++j) {
        if (std::find(treeEdges.begin(),
                      treeEdges.end(),
                      std::make_pair(i, j)) == treeEdges.end() and
            std::find(treeEdges.begin(),
                      treeEdges.end(),
                      std::make_pair(j, i)) == treeEdges.end()) {
          lostEdges.emplace_back(i, j);
        }
      }
    }

    // Sort lost edges by MI contribution
    std::sort(lostEdges.begin(), lostEdges.end(), [&](auto& a, auto& b) {
      return MI[a.first][a.second] > MI[b.first][b.second];
    });

    // Add back the top k% of missing edges
    size_t numEdgesToRestore = lostEdges.size() * (1 - kMinLostMI);
    std::cout << "Restoring " << numEdgesToRestore << " edges\n";
    for (size_t k = 0; k < numEdgesToRestore; ++k) {
      treeEdges.push_back(lostEdges[k]);
      totalMutualInformation += MI[lostEdges[k].first][lostEdges[k].second];
    }
  }

  return {treeEdges, totalMutualInformation, fullGraphMutualInformation};
}

// Function to extract pairwise conditionals using GTSAM's elimination
// Function to extract pairwise conditionals from a GaussianConditional
GaussianFactorGraph extractPairwiseConditionals(
    const GaussianConditional& fullConditional,
    const std::vector<std::pair<size_t, size_t>>& extractedEdges,
    KeyVector keys) {
  GaussianFactorGraph pairwiseConditionals;

  // Extract information matrix and mean vector from the full conditional
  // Matrix H = fullConditional.information();
  // Extract R matrix
  Matrix R = fullConditional.R();

  // Compute Information Matrix (Lambda = R^T * R)
  Matrix H = R.transpose() * R;
  Vector d = fullConditional.d();  // Mean vector

  // Get number of variables
  size_t dim = H.rows() / fullConditional.keys().size();

  for (const auto& edge : extractedEdges) {
    size_t i = edge.first;
    size_t j = edge.second;

    // Extract the submatrices
    Matrix H_ii = H.block(i * dim, i * dim, dim, dim);
    Matrix H_jj = H.block(j * dim, j * dim, dim, dim);
    Matrix H_ij = H.block(i * dim, j * dim, dim, dim);
    Vector d_i = d.segment(i * dim, dim);

    // Compute pairwise conditional parameters
    Matrix H_jj_inv = H_jj.inverse();  // Inverse for conditioning
    Matrix R = H_ii - H_ij * H_jj_inv * H_ij.transpose();
    Vector d_i_given_j = d_i - H_ij * H_jj_inv * d.segment(j * dim, dim);

    // Create the noise model from the diagonal of R
    Vector diagR = R.diagonal();
    auto noiseModel = noiseModel::Diagonal::Sigmas(diagR);

    // Create the GaussianConditional using diagonal noise
    auto pairwiseConditional =
        GaussianConditional::shared_ptr(new GaussianConditional(
            keys.at(i), d_i_given_j, R, keys.at(j), H_ij, noiseModel));
    pairwiseConditionals.push_back(pairwiseConditional);
  }
  return pairwiseConditionals;
}

void applyChowLiuApproximation(
    Eigen::Ref<Eigen::MatrixXd>& R,  // Explicit reference type
    const std::vector<std::pair<size_t, size_t>>& chowLiuEdges,
    size_t dim) {
  size_t totalVars = R.rows() / dim;

  // Create a lookup set for extracted edges
  std::set<std::pair<size_t, size_t>> edgeSet(chowLiuEdges.begin(),
                                              chowLiuEdges.end());

  // Zero out non-selected edges in the Hessian
  for (size_t i = 0; i < totalVars; ++i) {
    for (size_t j = 0; j < totalVars; ++j) {
      if (i != j && (edgeSet.find({i, j}) == edgeSet.end() and
                     edgeSet.find({j, i}) == edgeSet.end())) {
        R.block(i * dim, j * dim, dim, dim).setZero();
      }
    }
  }
}

void printMatrixDensity(const Eigen::MatrixXd& mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      std::cout << (mat(i, j) == 0.0 ? '0' : '*');
    }
    std::cout << '\n';
  }
}

GaussianBayesTree::Clique::shared_ptr buildNewClique(
    const GaussianBayesTree::Clique& oldClique,
    const Matrix& Ab) {
  auto conditional = oldClique.conditional();

  // Create a new GaussianConditional with modified R
  std::vector<size_t> dims;
  for (auto _ : conditional->keys()) {
    dims.push_back(Pose3::dimension);
  }
  dims.push_back(1);
  // print the size of the Ab matrix
  std::cout << fmt::format("Ab size: {}x{}\n", Ab.rows(), Ab.cols());
  // print the dims
  std::cout << "dims: ";
  for (auto d : dims) {
    std::cout << d << " ";
  }
  std::cout << "\n";
  VerticalBlockMatrix AbMatrix(dims, Ab);
  GaussianConditional::shared_ptr newConditional =
      boost::make_shared<GaussianConditional>(
          conditional->keys(), conditional->nrFrontals(), AbMatrix);

  GaussianBayesTreeClique::shared_ptr newClique =
      boost::make_shared<GaussianBayesTreeClique>(newConditional);
  return newClique;
}

// Main function to process a clique and compute Chow-Liu tree probabilities
GaussianBayesTreeClique::shared_ptr processClique(
    const GaussianBayesTree::Clique::shared_ptr& clique,
    size_t poseDim) {
  // Extract conditional
  auto conditional = clique->conditional();

  // Determine the number of poses
  size_t numPoses = conditional->keys().size();

  // Build Chow-Liu Tree for pose-wise structure
  auto chowLiuTree =
      buildChowLiuTree(conditional->information(), numPoses, poseDim);

  std::cout << fmt::format("Clique: \n  {}\n", *clique);

  // Print Chow-Liu Tree edges
  std::cout << "Chow-Liu Tree edges (Pose-wise):\n";

  auto keys = conditional->keys();
  for (const auto& edge : chowLiuTree.treeEdges) {
    std::cout << fmt::format(
        "({} -> {}) ", keys[edge.first], keys[edge.second]);
  }
  std::cout << "\n";
  std::cout << fmt::format("CLT/Full MI: {:.3f} / {:.3f}\n",
                           chowLiuTree.totalMutualInformation,
                           chowLiuTree.fullGraphMutualInformation);

  auto CLTBayesNet = extractPairwiseConditionals(
      *conditional, chowLiuTree.treeEdges, conditional->keys());

  // print the BayesNet
  std::cout << "Pairwise conditionals from Chow-Liu Tree:\n";
  for (const auto& factor : CLTBayesNet) {
    GaussianConditional::shared_ptr gaussianConditional =
        boost::dynamic_pointer_cast<GaussianConditional>(factor);
    std::cout << fmt::format("  {}\n", *gaussianConditional);
  }
  Matrix Ab = conditional->augmentedJacobianUnweighted();
  // get ref block of R matrix
  Eigen::Ref<Matrix> R = Ab.block(0, 0, Ab.rows(), conditional->R().cols());
  applyChowLiuApproximation(R, chowLiuTree.treeEdges, poseDim);

  // rebuild the clique
  return buildNewClique(*clique, Ab);
}

int main() {
  auto [graph, values] = readG2o("sphere2500.g2o", true);
  // add prior to first pose
  graph->add(PriorFactor<Pose3>(0, Pose3(), noiseModel::Unit::Create(6)));
  // optimize
  GaussNewtonOptimizer optimizer(*graph, *values);
  *values = optimizer.optimize();

  MarginalsExposeBayesTree marginals(*graph, *values);
  auto bayesTree = marginals.getBayesTree();

  std::map<GaussianBayesTreeClique::shared_ptr,
           GaussianBayesTreeClique::shared_ptr>
      sparsified_cliques;
  for (const auto& [_, clique] : bayesTree.nodes()) {
    if (!clique) continue;
    if (not sparsified_cliques.contains(clique)) {
      auto newClique = processClique(clique, 6);
      newClique->print("new clique");
      sparsified_cliques[clique] = newClique;
    }
  }

  // build the sparsified tree
  GaussianBayesTree sparseBayesTree;
  for (const auto& root : bayesTree.roots()) {
    CHECK(sparsified_cliques.contains(root),
          "Root clique not found in sparsified cliques");
    sparseBayesTree.insertRoot(sparsified_cliques[root]);
  }
  for (const auto& [clique, newClique] : sparsified_cliques) {
    for (const auto& child : clique->children) {
      CHECK(sparsified_cliques.contains(child),
            "Child clique not found in sparsified cliques");
      newClique->children.push_back(sparsified_cliques[child]);
    }
  }

  // print the sparsified tree
  sparseBayesTree.print("Sparsified Bayes Tree");

  marginals.setBayesTree(sparseBayesTree);

  // Compute marginals for all poses
  auto start = std::chrono::high_resolution_clock::now();
  auto cov = marginals.marginalCovariance(values->keySet());
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken CLT: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms\n";

  // compute marginals with the gtsam built-in marginals
  Marginals marginals_baseline(*graph, *values);
  start = std::chrono::high_resolution_clock::now();
  auto cov_bl = marginals_baseline.marginalCovariance(values->keySet());
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken Baseline: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms\n";
}