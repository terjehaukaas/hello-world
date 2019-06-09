

# ------------------------------------------------------------------------
# PACKAGES
# ------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm, lognorm, uniform
import terjesfunctions as fcns


# ------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------

# Distribution types (Normal, Lognormal, and Uniform are currently available)
distributions = ["Lognormal", "Lognormal", "Uniform"]

# Means
means = np.array([500.0, 2000.0, 5.0])

# Standard deviations
stdvs = np.array([100.0, 400.0, 0.5])

# Correlation
correlation = np.array([(1, 2, 0.3),
                        (1, 3, 0.2),
                        (2, 3, 0.2)])

# Limit-state function
evaluationCount = 0
def g(x):

    # Count the total number of evaluations
    global evaluationCount
    evaluationCount += 1

    # Get the value of the random variables from the input vector
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    g = 1.0 - x2 / (1000.0 * x3) - (x1 / (200.0 * x3))**2

    return g


# Derivative of the limit-state function by the "direct differentiation method" (DDM)
def dgddm(x):

    # Initialize the gradient vector
    dgdx = np.zeros(len(x))

    # Get the value of the random variables from the input vector
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # Calculate derivatives
    dgdx[0] = - 2 * x1 / (200.0 * x3)**2

    dgdx[1] = - 1 / (1000.0 * x3)

    dgdx[2] = x2 / (1000.0 * x3**2) + 2 * x1**2 / (200.0**2 * x3**3)

    return dgdx


# ------------------------------------------------------------------------
# ALGORITHM PARAMETERS
# ------------------------------------------------------------------------

ddm = True
maxSteps = 50
convergenceCriterion1 = 0.001
convergenceCriterion2 = 0.001
merit_y = 1.0
merit_g = 5.0


# ------------------------------------------------------------------------
# MODIFY THE CORRELATION MATRIX AND COMPUTE THE CHOLESKY DECOMPOSITION
# ------------------------------------------------------------------------

# Cholesky decomposition of the correlation matrix that is
# modified according to the Nataf the distribution
if 'correlation' in locals():

    print('\n'"Modifying correlation matrix...", flush=True)

    R0 = fcns.modifyCorrelationMatrix(means, stdvs, distributions, correlation)

    print('\n'"Done modifying correlation matrix...", flush=True)

    L = np.linalg.cholesky(R0)

else:
    L = np.identity(len(means))


# ------------------------------------------------------------------------
# FORM ALGORITHM
# ------------------------------------------------------------------------

# Set start point in the standard normal space
numRV = len(means)
y = np.zeros(numRV)

# Initialize the vectors
x = np.zeros(numRV)
dGdy = np.zeros(numRV)

# Start the FORM loop
for i in range(1, maxSteps+1):

    # Transform from y to x
    [x, dxdy] = fcns.transform_y_to_x(L, y, means, stdvs, distributions, True)

    # Evaluate the limit-state function, g(x) = G(y)
    gValue = g(x)

    # Evaluate the gradient in the x-space by finite difference, dg / dx
    if ddm:
        dgdx = dgddm(x)
    else:
        dgdx = fcns.dg(x, gValue, g)

    # Notice that dG/dy = dg/dx * dx/dz * dz/dy can be multiplied in opposite order if transposed
    dGdy = dxdy.dot(dgdx)

    # Determine the alpha-vector
    alpha = np.multiply(dGdy, -1 / np.linalg.norm(dGdy))

    # Calculate the first convergence criterion
    if i == 1:
        gfirst = gValue
    criterion1 = np.abs(gValue / gfirst)

    # Calculate the second convergence criterion
    yNormOr1 = np.linalg.norm(y)
    if yNormOr1 < 1.0:
        yNormOr1 = 1.0
    u_scaled = np.multiply(y, 1.0/yNormOr1)
    u_scaled_times_alpha = u_scaled.dot(alpha)
    criterion2 = np.linalg.norm(np.subtract(u_scaled, np.multiply(alpha, u_scaled_times_alpha)))

    # Print status
    print('\n'"FORM Step:", i, "Check1:", criterion1, ", Check2:", criterion2)

    # Check convergence
    if criterion1 < convergenceCriterion1 and criterion2 < convergenceCriterion2:

        # Here we converged; first calculate beta and pf
        betaFORM = np.linalg.norm(y)
        pfFORM = norm.cdf(-betaFORM)
        print('\n'"FORM analysis converged after", i, "steps with reliability index", betaFORM, "and pf", pfFORM)
        print('\n'"Total number of limit-state evaluations:", evaluationCount)
        print('\n'"Design point in x-space:", x)
        print('\n'"Design point in y-space:", y)

        # Importance vectors alpha and gamma
        print('\n'"Importance vector alpha:", alpha)
        dxdyProduct = dxdy.dot(np.transpose(dxdy))
        dydx = np.linalg.inv(dxdy)
        D_prime = np.zeros((numRV, numRV))
        for j in range(numRV):
            D_prime[j, j] = np.sqrt(dxdyProduct[j, j])
        gamma = (D_prime.dot(dydx)).dot(alpha)
        print('\n'"Importance vector gamma:", gamma)
        break

    # Take a step if convergence did not occur
    else:

        # Give message if non-convergence in maxSteps
        if i == maxSteps:
            print('\n'"FORM analysis did not converge")
            break

        # Determine the search direction
        searchDirection = np.multiply(alpha, (gValue / np.linalg.norm(dGdy) + alpha.dot(y))) - y

        # Define the merit function for the line search along the search direction
        def meritFunction(stepSize):

            # Determine the trial point corresponding to the step size
            yTrial = y + stepSize * searchDirection

            # Calculate the distance from the origin
            yNormTrial = np.linalg.norm(yTrial)

            # Transform from y to x
            xTrial = fcns.transform_y_to_x(L, yTrial, means, stdvs, distributions, False)

            # Evaluate the limit-state function
            gTrial = g(xTrial)

            # Evaluate the merit function m = 1/2 * ||y||^2  + c*g
            return merit_y * yNormTrial + merit_g * np.abs(gTrial/gfirst)

        # Perform the line search for the optimal step size
        stepSize = fcns.goldenSectionLineSearch(meritFunction, 0, 1.5, 50, 0.1, 0.1)
        #stepSize = fcns.armijoLineSearch(meritFunction, 1.0, 5)
        #stepSize = 1.0

        # Take a step
        y += stepSize * searchDirection

    # Increment the loop counter
    i += 1


