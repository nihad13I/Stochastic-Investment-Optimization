import jax
import jax.numpy as jnp
import numpy as np

# Parameters

theta = 0.7946314313301023
delta = 0.0785096072581796
gamma1 = 0.938491719403288
wage = 1.0
alpha = 0.3
beta = 0.95

nu = 1. - (1. - alpha) * theta
power1 = ((1. - alpha) * theta / wage) ** (1. / nu)
omega = (power1 ** ((1. - alpha) * theta) - power1 * wage)

z_grid = jnp.array([0.95, 1.0, 1.05])
P_z = jnp.array([[0.9, 0.1, 0.0],
                 [0.1, 0.8, 0.1],
                 [0.0, 0.1, 0.9]])

num_k = 512
num_z = 3
total_states = num_k * num_z
k_min = 0.1
k_max = 10.0


# Model Functions

def profit(k, z):
    return z * omega * (k ** theta)


def investment_cost(k_current, k_next):
    i_t = (k_next / k_current) - (1.0 - delta)

    total_cost = (k_current * i_t) + (0.5 * gamma1 * k_current * (i_t ** 2))
    return total_cost


def dividends(k_current, k_next, z):
    pi = profit(k_current, z)
    cost = investment_cost(k_current, k_next)
    return pi - cost


# Pre-comptation

step = (k_max - k_min) / (num_k - 1)
k_grid = k_min + jnp.arange(num_k) * step

k_curr_mesh = k_grid[:, None, None]
k_next_mesh = k_grid[None, :, None]

z_mesh = z_grid[None, None, :]

D_tensor = dividends(k_curr_mesh, k_next_mesh, z_mesh)


def flatten_index(k_idx, z_idx):
    return k_idx * num_z + z_idx


def induced_R_pi(policy_indices):
    row_indices = jnp.arange(num_k)[:, None]
    z_indices = jnp.arange(num_z)[None, :]

    R_matrix = D_tensor[row_indices, policy_indices, z_indices]

    return R_matrix.flatten()


def solve_value_function(policy_indices):
    R_pi = induced_R_pi(policy_indices)

    A_matrix = jnp.eye(total_states)

    for m in range(num_z):
        for n in range(num_z):

            prob_z = P_z[m, n]

            if prob_z == 0:
                continue

            current_k_indices = jnp.arange(num_k)
            flat_sources = flatten_index(current_k_indices, m)

            target_k_indices = policy_indices[:, m]
            flat_targets = flatten_index(target_k_indices, n)

            A_matrix = A_matrix.at[flat_sources, flat_targets].add(-beta * prob_z)

    V_flat = jnp.linalg.solve(A_matrix, R_pi)

    return V_flat.reshape((num_k, num_z))


def policy_improvement(V):
    expected_V = V @ P_z.T
    Q_values = D_tensor + beta * expected_V[None, :, :]

    new_policy_indices = jnp.argmax(Q_values, axis=1)

    return new_policy_indices


# Main Execution Loop

policy_indices = jnp.zeros((num_k, num_z), dtype=int)

for iteration in range(1000):

    V_pi = solve_value_function(policy_indices)

    new_policy_indices = policy_improvement(V_pi)

    n_changes = jnp.sum(new_policy_indices != policy_indices)

    if n_changes == 0:
        print("Converged!")
        policy_indices = new_policy_indices
        break

    policy_indices = new_policy_indices

# Output Generation

optimal_k_next_low = k_grid[policy_indices[:, 0]]
optimal_k_next_med = k_grid[policy_indices[:, 1]]
optimal_k_next_high = k_grid[policy_indices[:, 2]]

output_data = np.column_stack([
    np.array(k_grid),
    np.array(optimal_k_next_low),
    np.array(optimal_k_next_med),
    np.array(optimal_k_next_high)
])

try:
    np.savetxt(
        'policy_stochastic.csv',
        output_data,
        delimiter=',',
        fmt=['%.10f', '%.10f', '%.10f', '%.10f']
    )
except IOError as e:
    print("ERROR!")