#pragma once

#include <gtsam/nonlinear/Marginals.h>

#include <queue>

using namespace gtsam;

// Abstract Cost Function Structure
struct CostFunction {
  virtual ~CostFunction() = default;

  /**
   * @brief Computes the cost of a given clique.
   * @param clique The clique whose cost needs to be evaluated.
   * @return double The computed cost.
   */
  virtual double computeCost(
      const GaussianBayesTree::Clique::shared_ptr& clique) const = 0;

  /**
   * @brief Returns the maximum allowable cost before stopping expansion.
   * @return double The cost threshold.
   */
  virtual double getThreshold() const = 0;
};

// Condition Number Cost Function (Normalized)
// struct ConditionNumberCost : public CostFunction {
//     double kappaMin;  // Minimum expected condition number
//     double kappaMax;  // Maximum expected condition number

//     explicit ConditionNumberCost(double minKappa = 1.0, double maxKappa =
//     1e6)
//         : kappaMin(minKappa), kappaMax(maxKappa) {}

//     double computeCost(const GaussianBayesTree::Clique::shared_ptr& clique)
//     const override {
//         if (!clique || !clique->conditional()) return 0.0;

//         // Extract Hessian (Information matrix)
//         Matrix H = clique->conditional()->information();

//         // Estimate λ_max using Frobenius norm
//         double lambda_max = H.norm();

//         // Estimate λ_min using minimum diagonal entry
//         double lambda_min = H.diagonal().minCoeff();
//         lambda_min = std::max(lambda_min, 1e-8);  // Prevent division by zero

//         // Compute the condition number
//         double kappa = lambda_max / lambda_min;

//         // Normalize to [0,1] using logarithmic scaling
//         double logKappa = std::log10(kappa);
//         double logMin = std::log10(kappaMin);
//         double logMax = std::log10(kappaMax);

//         double normalizedCost = (logKappa - logMin) / (logMax - logMin);
//         return std::clamp(normalizedCost, 0.0, 1.0);
//     }
// };

struct ConditionNumberCost : public CostFunction {
  double threshold;

  explicit ConditionNumberCost(double maxThreshold) : threshold(maxThreshold) {}

  /**
   * @brief Estimates the condition number of a clique.
   */
  double computeCost(
      const GaussianBayesTree::Clique::shared_ptr& clique) const override {
    if (!clique || !clique->conditional()) return 0;

    // Extract the Hessian (Information matrix)
    Matrix H =
        clique->conditional()->information();  // GTSAM’s information matrix

    // Estimate λ_max using Frobenius norm
    double lambda_max = H.norm();  // Fast upper bound

    // Estimate λ_min using minimum diagonal entry (cheap lower bound)
    double lambda_min = H.diagonal().minCoeff();
    lambda_min = std::max(lambda_min, 1e-8);  // Prevent division by zero
    double kappa = lambda_max / lambda_min;
    // take log of condition number
    double logKappa = std::log10(kappa);

    return logKappa;
  }

  /**
   * @brief Returns the predefined threshold for condition number stopping.
   */
  double getThreshold() const override { return threshold; }
};

// Frontal Variable Count Cost Function
struct FrontalVariableCountCost : public CostFunction {
  FrontalVariableCountCost(double maxThreshold = 1.0)
      : threshold(maxThreshold) {}

  double computeCost(
      const GaussianBayesTree::Clique::shared_ptr& clique) const override {
    if (!clique || !clique->conditional()) return 1.0;

    // Get number of frontal variables
    size_t numFrontals = clique->conditional()->nrFrontals();

    // Compute cost as 1 / (1 + numFrontals) to encourage deeper search
    return -static_cast<double>(numFrontals);
  }

  double getThreshold() const override { return 1.0; }

  double threshold;
};

// **Multi-Cost Function Wrapper**
class CombinedCostFunction : public CostFunction {
 public:
  std::vector<std::shared_ptr<CostFunction>> costFunctions;
  std::vector<double> weights;

  CombinedCostFunction(std::vector<std::shared_ptr<CostFunction>> costFuncs,
                       std::vector<double> costWeights,
                       double maxThreshold = 1.0)
      : costFunctions(std::move(costFuncs)),
        weights(std::move(costWeights)),
        threshold(maxThreshold) {}

  /**
   * @brief Computes the weighted sum of multiple cost functions.
   */
  double computeCost(
      const GaussianBayesTree::Clique::shared_ptr& clique) const override {
    double totalCost = 0.0;
    for (size_t i = 0; i < costFunctions.size(); i++) {
      totalCost += weights[i] * costFunctions[i]->computeCost(clique);
    }
    return totalCost;
  }

  double getThreshold() const override { return threshold; }

  double threshold;
};

// Subtree Search Class (Best-First)
class SubtreeSearch {
 public:
  /**
   * @brief Searches for the lowest-cost subtree within the threshold.
   * @param bayesTree The Bayes tree from GTSAM.
   * @param costFunction A shared pointer to a CostFunction instance.
   * @return The selected optimal subtree with the lowest total cost.
   */
  static auto searchOptimalSubtree(const GaussianBayesTree& bayesTree,
                                   std::shared_ptr<CostFunction> costFunction) {
    std::vector<GaussianBayesTree::Clique::shared_ptr> bestSubtree;
    double bestTotalCost = std::numeric_limits<double>::infinity();

    // Priority queue (min-heap) to explore low-cost cliques first
    using CliqueCostPair =
        std::pair<double, GaussianBayesTree::Clique::shared_ptr>;
    std::priority_queue<CliqueCostPair,
                        std::vector<CliqueCostPair>,
                        std::greater<>>
        pq;

    // Start from the root clique
    GaussianBayesTree::Clique::shared_ptr root = bayesTree.roots().front();
    pq.push({costFunction->computeCost(root), root});

    // Track visited cliques
    std::set<GaussianBayesTree::Clique::shared_ptr> visited;

    while (!pq.empty()) {
      auto [currentCost, clique] = pq.top();
      pq.pop();

      if (!clique || visited.count(clique)) continue;
      visited.insert(clique);

      std::vector<GaussianBayesTree::Clique::shared_ptr> candidateSubtree;
      double totalCost = 0.0;

      // Expand this subtree
      expandSubtree(clique, costFunction, candidateSubtree, totalCost);

      // Check if it's the best valid subtree
      if (totalCost <= costFunction->getThreshold() &&
          totalCost < bestTotalCost) {
        bestSubtree = candidateSubtree;
        bestTotalCost = totalCost;
      }

      // Push child cliques to explore next (greedy order)
      for (const auto& child : clique->children) {
        double childCost = costFunction->computeCost(child);
        pq.push({childCost, child});
      }
    }

    return std::tuple{bestSubtree, bestTotalCost};
  }

 private:
  /**
   * @brief Expands a clique to find its total cost and subtree.
   * @param clique The current clique being evaluated.
   * @param costFunction The function to compute the cost of a clique.
   * @param selectedSubtree The selected optimal subtree.
   * @param totalCost The accumulated cost.
   */
  static void expandSubtree(
      const GaussianBayesTree::Clique::shared_ptr& clique,
      std::shared_ptr<CostFunction> costFunction,
      std::vector<GaussianBayesTree::Clique::shared_ptr>& selectedSubtree,
      double& totalCost) {
    if (!clique) return;

    double cliqueCost = costFunction->computeCost(clique);
    totalCost += cliqueCost;

    if (totalCost <= costFunction->getThreshold()) {
      selectedSubtree.push_back(clique);

      // Recursively expand child cliques
      for (const auto& child : clique->children) {
        expandSubtree(child, costFunction, selectedSubtree, totalCost);
      }
    } else {
      totalCost -= cliqueCost;  // Rollback cost if stopping here
    }
  }
};