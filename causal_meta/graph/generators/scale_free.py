"""
Scale-free network generators for the causal_meta library.

This module provides implementations for generating scale-free network structures,
including the Barabási-Albert preferential attachment model and its variations.
"""
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
import random
import numpy as np

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.errors import GraphGenerationError


class ScaleFreeNetworkGenerator:
    """
    Generator class for scale-free network structures.

    This class provides methods for generating various types of scale-free networks,
    including Barabási-Albert model and its variants. Scale-free networks are
    characterized by a power-law degree distribution, where the probability of a
    node having k connections follows P(k) ~ k^(-γ), with γ typically between 2 and 3.
    """

    @staticmethod
    def barabasi_albert(
        num_nodes: int,
        m: int = 2,
        directed: bool = True,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a scale-free network using the Barabási-Albert preferential attachment model.

        The Barabási-Albert model creates a network where new nodes attach to existing nodes
        with a probability proportional to the existing nodes' degrees, leading to a
        power-law degree distribution.

        Args:
            num_nodes: Number of nodes in the network
            m: Number of edges to attach from each new node to existing nodes
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A scale-free network instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < m + 1:
            raise GraphGenerationError(
                f"Number of nodes must be at least m+1 ({m+1}). Got {num_nodes}.")
        if m < 1:
            raise GraphGenerationError("Parameter m must be at least 1.")
        if m >= num_nodes:
            raise GraphGenerationError(
                f"Parameter m ({m}) must be less than the number of nodes ({num_nodes}).")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add initial nodes to the graph
        for i in range(m + 1):
            graph.add_node(i)

        # Connect the initial nodes (create a complete graph of m+1 nodes)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                graph.add_edge(i, j)
                if not directed:
                    graph.add_edge(j, i)

        # Add remaining nodes using preferential attachment
        for i in range(m + 1, num_nodes):
            graph.add_node(i)
            # Calculate degrees for preferential attachment
            targets = ScaleFreeNetworkGenerator._preferential_attachment(
                graph, i, m, directed)

            # Add edges from the new node to the selected targets
            for target in targets:
                graph.add_edge(i, target)
                if not directed:
                    graph.add_edge(target, i)

        return graph

    @staticmethod
    def _preferential_attachment(
        graph: Union[CausalGraph, DirectedGraph],
        new_node: int,
        m: int,
        directed: bool
    ) -> List[int]:
        """
        Select nodes for attachment based on preferential attachment mechanism.

        Args:
            graph: The current graph
            new_node: The new node being added
            m: Number of edges to create
            directed: Whether the graph is directed

        Returns:
            List of target nodes selected for attachment
        """
        # Create a list of nodes with probability proportional to degree
        node_degrees = {}
        total_degree = 0
        candidates = list(range(new_node))  # Only consider existing nodes

        for node in candidates:
            # For directed graphs, use out-degree; for undirected, use total degree
            if directed:
                degree = len(graph.get_successors(node))
            else:
                degree = len(graph.get_successors(node)) + \
                    len(graph.get_predecessors(node))

            # Ensure all nodes have a non-zero probability (add 1 to all degrees)
            degree += 1

            node_degrees[node] = degree
            total_degree += degree

        # Select m distinct nodes based on their degree probability
        targets = []
        while len(targets) < m and candidates:
            # Use a more efficient selection algorithm:
            # Create a cumulative probability distribution and sample once
            r = random.uniform(0, total_degree)
            cumulative = 0

            for node, degree in node_degrees.items():
                if node in targets:
                    continue

                cumulative += degree
                if cumulative >= r:
                    targets.append(node)
                    # Remove the selected node from candidates and update total_degree
                    total_degree -= degree
                    node_degrees[node] = 0
                    break

            # Handle edge case if we didn't select a node due to rounding errors
            if len(targets) < m and len(targets) == len(set(targets)) and candidates:
                available = [n for n in candidates if n not in targets]
                if available:
                    selected = random.choice(available)
                    targets.append(selected)
                    # Update probabilities
                    total_degree -= node_degrees[selected]
                    node_degrees[selected] = 0

        return targets

    @staticmethod
    def barabasi_albert_with_alpha(
        num_nodes: int,
        m: int = 2,
        alpha: float = 1.0,
        directed: bool = True,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a scale-free network using a modified Barabási-Albert model with an
        adjustable preferential attachment exponent alpha.

        The probability of attaching to a node i is proportional to (degree_i)^alpha.
        When alpha=1, this reduces to the standard Barabási-Albert model.

        Args:
            num_nodes: Number of nodes in the network
            m: Number of edges to attach from each new node to existing nodes
            alpha: Preferential attachment exponent (default: 1.0)
                  - alpha > 1: Increases the "rich get richer" effect
                  - alpha < 1: Decreases the effect, making the degree distribution less skewed
                  - alpha = 0: Random attachment (equivalent to an Erdős–Rényi model)
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A scale-free network instance with the specified alpha parameter

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < m + 1:
            raise GraphGenerationError(
                f"Number of nodes must be at least m+1 ({m+1}). Got {num_nodes}.")
        if m < 1:
            raise GraphGenerationError("Parameter m must be at least 1.")
        if m >= num_nodes:
            raise GraphGenerationError(
                f"Parameter m ({m}) must be less than the number of nodes ({num_nodes}).")
        if alpha < 0:
            raise GraphGenerationError("Parameter alpha must be non-negative.")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add initial nodes to the graph
        for i in range(m + 1):
            graph.add_node(i)

        # Connect the initial nodes (create a complete graph of m+1 nodes)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                graph.add_edge(i, j)
                if not directed:
                    graph.add_edge(j, i)

        # Add remaining nodes using modified preferential attachment with alpha
        for i in range(m + 1, num_nodes):
            graph.add_node(i)

            # Calculate degrees for modified preferential attachment
            node_degrees = {}
            total_degree_alpha = 0
            candidates = list(range(i))  # Only consider existing nodes

            for node in candidates:
                # Calculate degree based on graph type
                if directed:
                    degree = len(graph.get_successors(node))
                else:
                    degree = len(graph.get_successors(node)) + \
                        len(graph.get_predecessors(node))

                # Add 1 to ensure non-zero probability
                degree += 1

                # Apply alpha exponent
                degree_alpha = degree ** alpha

                node_degrees[node] = degree_alpha
                total_degree_alpha += degree_alpha

            # Select m distinct nodes based on modified probability
            targets = []
            while len(targets) < m and candidates:
                r = random.uniform(0, total_degree_alpha)
                cumulative = 0

                for node, degree_alpha in node_degrees.items():
                    if node in targets:
                        continue

                    cumulative += degree_alpha
                    if cumulative >= r:
                        targets.append(node)
                        # Remove the selected node from candidates
                        total_degree_alpha -= degree_alpha
                        node_degrees[node] = 0
                        break

                # Handle edge case
                if len(targets) < m and len(targets) == len(set(targets)) and candidates:
                    available = [n for n in candidates if n not in targets]
                    if available:
                        selected = random.choice(available)
                        targets.append(selected)
                        # Update probabilities
                        total_degree_alpha -= node_degrees[selected]
                        node_degrees[selected] = 0

            # Add edges from the new node to the selected targets
            for target in targets:
                graph.add_edge(i, target)
                if not directed:
                    graph.add_edge(target, i)

        return graph

    @staticmethod
    def extended_barabasi_albert(
        num_nodes: int,
        m: int = 2,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        directed: bool = True,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a scale-free network using an extended Barabási-Albert model with
        additional parameters to control the network's properties.

        This extended model combines preferential attachment with random attachment
        and allows adjustment of the power-law exponent.

        Args:
            num_nodes: Number of nodes in the network
            m: Number of edges to attach from each new node to existing nodes
            alpha: Preferential attachment exponent (default: 1.0)
            beta: Probability of random attachment vs. preferential attachment (default: 0.0)
                  - beta=0: Pure preferential attachment (standard BA model)
                  - beta=1: Pure random attachment
                  - 0<beta<1: Mixture of both mechanisms
            gamma: Aging factor for older nodes (default: 0.0)
                  - gamma=0: No aging effect
                  - gamma>0: Older nodes become less attractive over time
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            An extended scale-free network instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < m + 1:
            raise GraphGenerationError(
                f"Number of nodes must be at least m+1 ({m+1}). Got {num_nodes}.")
        if m < 1:
            raise GraphGenerationError("Parameter m must be at least 1.")
        if m >= num_nodes:
            raise GraphGenerationError(
                f"Parameter m ({m}) must be less than the number of nodes ({num_nodes}).")
        if alpha < 0:
            raise GraphGenerationError("Parameter alpha must be non-negative.")
        if beta < 0 or beta > 1:
            raise GraphGenerationError(
                "Parameter beta must be between 0 and 1.")
        if gamma < 0:
            raise GraphGenerationError("Parameter gamma must be non-negative.")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add initial nodes to the graph
        for i in range(m + 1):
            graph.add_node(i)

        # Connect the initial nodes (create a complete graph of m+1 nodes)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                graph.add_edge(i, j)
                if not directed:
                    graph.add_edge(j, i)

        # Add remaining nodes using extended attachment model
        for i in range(m + 1, num_nodes):
            graph.add_node(i)

            # Initialize selection probabilities
            node_probabilities = {}
            total_probability = 0
            candidates = list(range(i))

            # Calculate attachment probabilities for each candidate node
            for node in candidates:
                # Determine if this selection will use random or preferential attachment
                if random.random() < beta:
                    # Random attachment: equal probability for all nodes
                    prob = 1.0
                else:
                    # Preferential attachment based on degree
                    if directed:
                        degree = len(graph.get_successors(node))
                    else:
                        degree = len(graph.get_successors(node)) + \
                            len(graph.get_predecessors(node))

                    # Add 1 to ensure non-zero probability
                    degree += 1

                    # Apply alpha exponent for preferential attachment
                    prob = degree ** alpha

                    # Apply aging effect if gamma > 0
                    if gamma > 0:
                        # Calculate node age (newer nodes have lower age)
                        age = i - node
                        # Apply aging factor: older nodes become less attractive
                        aging_factor = np.exp(-gamma * age)
                        prob *= aging_factor

                node_probabilities[node] = prob
                total_probability += prob

            # Select m distinct nodes based on calculated probabilities
            targets = []
            while len(targets) < m and candidates:
                r = random.uniform(0, total_probability)
                cumulative = 0

                for node, prob in node_probabilities.items():
                    if node in targets:
                        continue

                    cumulative += prob
                    if cumulative >= r:
                        targets.append(node)
                        # Remove the selected node from candidates
                        total_probability -= prob
                        node_probabilities[node] = 0
                        break

                # Handle edge case
                if len(targets) < m and len(targets) == len(set(targets)) and candidates:
                    available = [n for n in candidates if n not in targets]
                    if available:
                        selected = random.choice(available)
                        targets.append(selected)
                        # Update probabilities
                        total_probability -= node_probabilities[selected]
                        node_probabilities[selected] = 0

            # Add edges from the new node to the selected targets
            for target in targets:
                graph.add_edge(i, target)
                if not directed:
                    graph.add_edge(target, i)

        return graph

    @staticmethod
    def verify_power_law(
        graph: Union[CausalGraph, DirectedGraph],
        directed: bool = True,
        min_nodes: int = 100,
        confidence_interval: bool = False
    ) -> Dict[str, float]:
        """
        Verify if a graph follows a power-law degree distribution and estimate the exponent.

        Args:
            graph: The graph to analyze
            directed: Whether the graph is directed
            min_nodes: Minimum number of nodes required for reliable estimation
            confidence_interval: Whether to compute confidence intervals for the exponent

        Returns:
            Dictionary containing:
                - 'exponent': Estimated power-law exponent
                - 'r_squared': Coefficient of determination (goodness of fit)
                - 'p_value': P-value for the fit (if confidence_interval is True)
                - 'conf_low', 'conf_high': Confidence interval bounds (if confidence_interval is True)

        Raises:
            GraphGenerationError: If the graph has fewer nodes than min_nodes
        """
        # Check if the graph has enough nodes for reliable estimation
        num_nodes = len(graph._nodes)
        if num_nodes < min_nodes:
            raise GraphGenerationError(
                f"Graph has {num_nodes} nodes, but at least {min_nodes} are needed for reliable power-law estimation.")

        # Calculate degree distribution
        degrees = []
        for node in range(num_nodes):
            if directed:
                # For directed graphs, use in-degree (number of predecessors)
                degree = len(graph.get_predecessors(node))
            else:
                # For undirected graphs, use total degree
                degree = len(graph.get_successors(node)) + \
                    len(graph.get_predecessors(node))
                # Adjust for double-counting in undirected graphs
                if not directed:
                    degree = degree // 2

            degrees.append(degree)

        # Count frequency of each degree
        degree_counts = {}
        for degree in degrees:
            if degree not in degree_counts:
                degree_counts[degree] = 0
            degree_counts[degree] += 1

        # Prepare data for log-log plot (P(k) vs k)
        x_values = []  # log(degree)
        y_values = []  # log(frequency)

        # Use complementary cumulative distribution function (CCDF) for better power-law fit
        total_nodes = len(degrees)
        sorted_degrees = sorted(degree_counts.keys())

        for degree in sorted_degrees:
            if degree == 0:
                continue  # Skip degree 0 for log calculations

            # Calculate the CCDF: P(X > x)
            ccdf = sum(degree_counts[k]
                       for k in degree_counts if k >= degree) / total_nodes

            if ccdf > 0:  # Avoid log(0)
                x_values.append(np.log(degree))
                y_values.append(np.log(ccdf))

        # Calculate power-law exponent using linear regression on log-log plot
        if len(x_values) < 2:
            raise GraphGenerationError(
                "Not enough unique degrees to estimate power-law exponent.")

        # Convert to numpy arrays
        x = np.array(x_values)
        y = np.array(y_values)

        # Simple linear regression
        n = len(x)
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate R-squared (coefficient of determination)
        y_pred = slope * x + intercept
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # For CCDF fit, the power-law exponent is gamma = 1 - slope
        # For PDF fit, it would be gamma = -(slope)
        gamma = 1 - slope

        result = {
            'exponent': gamma,
            'r_squared': r_squared
        }

        # Compute confidence intervals if requested
        if confidence_interval:
            try:
                import scipy.stats as stats

                # Standard error of the slope
                x_mean = np.mean(x)
                x_var = np.sum((x - x_mean) ** 2)
                y_pred = slope * x + intercept
                residuals = y - y_pred
                s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
                se_slope = s_err / np.sqrt(x_var)

                # Calculate t-value for 95% confidence interval
                t_value = stats.t.ppf(0.975, n - 2)

                # Calculate confidence interval for slope
                slope_ci_low = slope - t_value * se_slope
                slope_ci_high = slope + t_value * se_slope

                # Convert slope confidence interval to gamma confidence interval
                gamma_ci_low = 1 - slope_ci_high
                gamma_ci_high = 1 - slope_ci_low

                # Calculate p-value
                t_stat = slope / se_slope
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                result['p_value'] = p_value
                result['conf_low'] = gamma_ci_low
                result['conf_high'] = gamma_ci_high
            except ImportError:
                # If scipy is not available, skip confidence interval calculation
                pass

        return result
