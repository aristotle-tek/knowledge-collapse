import numpy as np
from scipy.stats import truncnorm, norm, uniform
from scipy.stats import lognorm, gaussian_kde
from scipy.stats import t, rv_continuous
#import matplotlib.pyplot as plt

from scipy.integrate import simps, quad



def draw_normal(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


def draw_t(df, n, rng=None):
    """
    Generate random samples from a Student's t-distribution.

    Parameters:
    df (float): Degrees of freedom.
    n (int): Number of samples to generate.
    rng (numpy.random.Generator, optional): If None, use default generator

    Return np.ndarray

    """
    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2 for the variance to be defined.")
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_t(df, size=n)


def draw_t_scaled(df, n, std_dev, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.standard_t(df, size=n)
    # Scale the samples to achieve the desired standard deviation
    # For df > 2, the standard deviation of a standard t-distribution is sqrt(df / (df - 2))
    # Therefore, to scale to a desired std_dev, multiply by the ratio of the desired std_dev to the standard one.
    if df > 2:
        scale_factor = std_dev * np.sqrt((df - 2) / df)
        scaled_samples = samples * scale_factor
    else:
        raise ValueError("Degrees of freedom must be greater than 2 for the variance to be defined.")
    return scaled_samples



def draw_truncated_scaled_t(
        df, sigma_tr=None, n=100, std_dev=1.0, 
        lower_bound=None, upper_bound=None, rng=None):

    # could make more efficient but meh.

    if rng is None:
        rng = np.random.default_rng()

    # If lower_bound and upper_bound are not specified, use -/+ sigma_tr
    if lower_bound is None and upper_bound is None:
        if sigma_tr is None:
            raise ValueError("Either sigma_tr or both lower_bound and upper_bound must be specified.")
        lower_bound = -sigma_tr
        upper_bound = sigma_tr
    elif lower_bound is None or upper_bound is None:
        raise ValueError("Both lower_bound and upper_bound must be specified if one is.")

    samples = []
    while len(samples) < n:
        # Draw a batch of samples from the t-distribution
        batch = draw_t_scaled(df, n*10, std_dev, rng) # rng.standard_t(df, size=n*10)  # Use rng to generate samples
        
        # Filter samples within the truncation range
        truncated = batch[(batch >= lower_bound) & (batch <= upper_bound)]
        
        samples.extend(truncated.tolist())
        if len(samples) >= n:  # Break if we have enough samples
            break
    
    return np.array(samples[:n])



def draw_truncated_t(df, sigma_tr=None, n=100, lower_bound=None, upper_bound=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # If lower_bound and upper_bound are not specified, use -/+ sigma_tr
    if lower_bound is None and upper_bound is None:
        if sigma_tr is None:
            raise ValueError("Either sigma_tr or both lower_bound and upper_bound must be specified.")
        lower_bound = -sigma_tr
        upper_bound = sigma_tr
    elif lower_bound is None or upper_bound is None:
        raise ValueError("Both lower_bound and upper_bound must be specified if one is.")

    samples = []
    while len(samples) < n:
        # Draw a batch of samples from the t-distribution
        batch = rng.standard_t(df, size=n*10)  # Use rng to generate samples
        
        # Filter to those within the truncation range
        truncated = batch[(batch >= lower_bound) & (batch <= upper_bound)]
        
        samples.extend(truncated.tolist())
        if len(samples) >= n:  # Break if we have enough samples
            break
    
    return np.array(samples[:n])



def draw_log_normal(mu, sigma, size=1):
    """
    Draw (size) samples from a log-normal distribution given the mean (mu) and standard deviation (sigma).
    Return np.ndarray
    """
    # The mean and standard deviation of the underlying normal distribution
    mean_normal = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
    sigma_normal = np.sqrt(np.log(1 + (sigma**2 / mu**2)))
    
    samples = np.random.lognormal(mean=mean_normal, sigma=sigma_normal, size=size)
    return samples



def initialize_kde_pdf(samples):
    # ensure np.array
    samples = np.asarray(samples)

    if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
        raise ValueError("Samples must not contain NaNs or Infs.")

    kde = gaussian_kde(samples)
    return kde, samples



def initialize_flat_kde_pdf(mean=0, std_dev=10, n_samples=10000):
    samples = np.random.normal(mean, std_dev, n_samples)
    return gaussian_kde(samples), samples



def initialize_pdf(distribution_type, params):
    if distribution_type == "uniform":
        a, b = params
        return uniform(loc=a, scale=b-a), np.NaN
    elif distribution_type == "kde_t_dist":
        df, n = params
        samples = draw_t(df, n)
        return initialize_kde_pdf(samples)
    elif distribution_type == "kde_normal_nonzeromean":
        mean, n = params
        samples = np.random.normal(mean, 1, n)
        return initialize_kde_pdf(samples)
    elif distribution_type == "kde_standard_normal":
        n = params
        samples = np.random.normal(0, 1, n)
        return initialize_kde_pdf(samples)
    else:
        raise ValueError("Invalid distr type.")



def update_kde_pdf(original_kde, new_samples):
    # ensure np.array
    new_samples = np.asarray(new_samples)

    if np.any(np.isnan(new_samples)) or np.any(np.isinf(new_samples)):
        raise ValueError("New samples must not contain NaNs or Infs.")

    if new_samples.ndim == 1:
        new_samples = new_samples.reshape(-1, 1) # Reshape to 2D if 1D

    combined_samples = np.concatenate([original_kde.dataset, new_samples.T], axis=1)

    updated_kde = gaussian_kde(combined_samples)
    return updated_kde



def kde_pdf_estimate_mn_std_bounds(kde_pdf, sigma_tr, eval_points=10000, eval_range=10):
    # For KDE and similar, approximate mean and std, return with lb, upper bd
    x = np.linspace(-eval_range, eval_range, eval_points)
    pdf_values = kde_pdf(x)
    dx = x[1] - x[0]

    mean = simps(x * pdf_values, x) / simps(pdf_values, x)
    std = np.sqrt(simps((x - mean)**2 * pdf_values, x) / simps(pdf_values, x))
    
    lower_bound = mean - sigma_tr * std
    upper_bound = mean + sigma_tr * std
    return mean, std, lower_bound, upper_bound



def check_integration_error(
        public_pdf, lower_tail_bounds, upper_tail_bounds, center_bounds):
    """
    Check the error in the integration of a PDF by summing up the area under the PDF 
    for three regions and comparing the sum to 1.

    Parameters:
    public_pdf (gaussian_kde object): The PDF represented by gaussian_kde.
    lower_tail_bounds (list): The bounds for the lower tail region.
    upper_tail_bounds (list): The bounds for the upper tail region.
    center_bounds (list): The bounds for the center region.

    Returns:
    float: The error in the integration, calculated as the absolute difference from 1.
    """
    # Integrate over the three regions
    lower_tail_area, _ = quad(public_pdf, lower_tail_bounds[0], lower_tail_bounds[1])
    upper_tail_area, _ = quad(public_pdf, upper_tail_bounds[0], upper_tail_bounds[1])
    center_area, _ = quad(public_pdf, center_bounds[0], center_bounds[1])

    total_area = lower_tail_area + upper_tail_area + center_area
    error = abs(1 - total_area)

    return error



def calculate_pdf_std(pdf, eval_range=10, eval_points=10000):
    if hasattr(pdf, 'std'):
        return pdf.std()
    else:
        # For KDEs or similar, approximate the standard deviation
        x = np.linspace(-eval_range, eval_range, eval_points)
        pdf_values = pdf(x)
        mean = simps(x * pdf_values, x) / simps(pdf_values, x)  # Approximate mean
        std_squared = simps((x - mean)**2 * pdf_values, x) / simps(pdf_values, x)
        return np.sqrt(std_squared)



def kl_divergence_kde_normal(kde, standard_normal, lower_bound=-10, upper_bound=10):
    def integrand(x):
        pdf_kde = np.exp(kde.logpdf(x))
        pdf_normal = standard_normal.pdf(x)
        return pdf_kde * np.log(pdf_kde / pdf_normal)

    return quad(integrand, lower_bound, upper_bound)[0]



def calculate_hellinger_distance(public_pdf, t_dist_pdf, lower_bound, upper_bound):
    """
    Calculate the Hellinger Distance between two probability density functions (PDFs).

    Parameters:
    t_dist_pdf (function): The PDF of the theoretical t-distribution. (use t_distribution_pdf = lambda x: t.pdf(x, df) )
    public_pdf (gaussian_kde object): The PDF represented by gaussian_kde.
    lower_bound (float)
    upper_bound (float)

    Returns:
    float: The Hellinger Distance between the two PDFs.
    """
    # Define the Hellinger integrand
    hellinger_integrand = lambda x: ((np.sqrt(public_pdf(x)) - np.sqrt(t_dist_pdf(x)))**2)

    # Use numerical integration
    hellinger_distance, _ = quad(hellinger_integrand, lower_bound, upper_bound)

    # Normalize the Hellinger distance
    return (1/np.sqrt(2)) * np.sqrt(hellinger_distance)



def calculate_approximate_kl_divergence(
        t_dist_pdf, public_pdf, lower_bound, 
        upper_bound, num_points=1000):
    """
    Calculate an approximate Kullback-Leibler (KL) Divergence between two probability density functions (PDFs).

    Parameters:
    t_dist_pdf (function): The PDF of the theoretical t-distribution.
    public_pdf (gaussian_kde object): The PDF represented by gaussian_kde.
    lower_bound (float)
    upper_bound (float)
    num_points (int): Number of points to use in the approximation.

    Returns:
    float: The approximate KL Divergence between the two PDFs.
    """
    x = np.linspace(lower_bound, upper_bound, num_points)
    
    # Evaluate both PDFs at these points
    t_dist_values = t_dist_pdf(x)
    public_pdf_values = public_pdf(x)

    # Avoid division by zero or log of zero by adding a small constant
    epsilon = 1e-10
    t_dist_values[t_dist_values < epsilon] = epsilon
    public_pdf_values[public_pdf_values < epsilon] = epsilon

    # Calculate the KL divergence using the discrete approximation
    kl_divergence = np.sum(public_pdf_values * np.log(public_pdf_values / t_dist_values)) * (x[1] - x[0])

    return kl_divergence



def calc_individ_expected_values_fullsample(theta_i, avg_full_innov):
    full_draw_value = theta_i * avg_full_innov
    return full_draw_value


def calc_individ_expected_values_truncated(theta_i, avg_truncated_innov):
    truncated_value = theta_i * avg_truncated_innov
    return truncated_value



vectorized_calc_benefits_fn_fullsamp = np.vectorize(calc_individ_expected_values_fullsample)
vectorized_calc_benefits_fn_truncated = np.vectorize(calc_individ_expected_values_truncated)




def calc_hellinger_innovation_pdf_regions(
        old_pdf, new_pdf, true_pdf, lower_bound, upper_bound):
    """
    {Improvement} = D(\text{PDF}_{\text{public_t-1}}, \text{PDF}_{\text{true}}) - D({\text{public_t}}, \text{PDF}_{\text{true}})
    - Need to include lower, upper bounds because use approx - numerical integration
    """
    old_true_dist = calculate_hellinger_distance(old_pdf, true_pdf, lower_bound, upper_bound)
    new_true_dist = calculate_hellinger_distance(new_pdf, true_pdf, lower_bound, upper_bound)
    return old_true_dist - new_true_dist



def decide_choice_from_values(
        value_estimates_fullsamp, value_estimates_truncated,
        fullsample_cost, discount):

    """
    Decides between buying a full sample, buying a truncated sample, or not buying at all.
    Applies a utility decision-making rule to maximize the difference between utility and cost.
    Works element-wise for each individual in the dataset.
    
    Returns a NumPy array where:
    2 indicates choosing the full sample product,
    1 indicates choosing the truncated product,
    0 indicates not buying.
    """

    net_benefit_fullsamp = value_estimates_fullsamp - fullsample_cost
    net_benefit_truncated = value_estimates_truncated - (fullsample_cost * discount)
    
    results = np.zeros_like(value_estimates_fullsamp)
    
    # Conditions for choosing each option
    choose_fullsample = (net_benefit_fullsamp > net_benefit_truncated) & (net_benefit_fullsamp > 0)
    choose_truncated = (net_benefit_truncated > net_benefit_fullsamp) & (net_benefit_truncated > 0)
    choose_none = (net_benefit_fullsamp <= 0) & (net_benefit_truncated <= 0)
    
    # If net benefits are equal and positive, favor choosing truncated, according to given preference
    equal_net_benefit_positive = (net_benefit_fullsamp == net_benefit_truncated) & (net_benefit_truncated > 0)
    choose_truncated = choose_truncated | equal_net_benefit_positive

    results[choose_fullsample] = 2
    results[choose_truncated] = 1

    return results


def update_truncation_limits(sigma_tr, curr_stdev, true_stdev):
    truncation_as_fraction_of_true_t_stdev = sigma_tr / true_stdev

    return truncation_as_fraction_of_true_t_stdev * curr_stdev



def draw_est_samples_based_on_choices(
        choice_vector, df, sigma_tr, st_dev, 
        lower_bound=None, upper_bound=None, setseed=None):

    # Draw from current distribution (possibly with a specific st_dev)
    if setseed is not None:
        rng = np.random.default_rng(np.random.PCG64(setseed))
    else:
        rng = np.random.default_rng()
    
    n_truncated_samples = sum([x == 1 for x in choice_vector])
    n_full_samples = sum([x == 2 for x in choice_vector])

    if st_dev is None:
            tr_samples = draw_truncated_t(df, sigma_tr, n_truncated_samples, 
                lower_bound=lower_bound, upper_bound=upper_bound, rng=rng)
            full_samples = draw_t(df, n_full_samples, rng=rng)
    else:
        tr_samples =  draw_truncated_scaled_t(
            df, sigma_tr=sigma_tr, n=n_truncated_samples, std_dev=st_dev,
            lower_bound=lower_bound, upper_bound=upper_bound, rng=rng
            )
        full_samples =  draw_t_scaled(df, n_full_samples, st_dev, rng=rng)

    return tr_samples, full_samples




def calc_innov_for_samples(
        samples, public_pdf, t_distribution_pdf, 
        infinite_approx_for_tails=20, enforce_nonneg=True):

    # Calculate the change in Hellinger distance towards the true distribution for each sample
    innovs = []
    for sample in samples:
        temp = update_kde_pdf(public_pdf, [sample])
        innov = calc_hellinger_innovation_pdf_regions(public_pdf, temp,
            t_distribution_pdf, -infinite_approx_for_tails, infinite_approx_for_tails)
        if enforce_nonneg:
            innov = max(0,innov)
        innovs.append(100*innov)
    return innovs



def update_value_estim(hist_list, lr=0.1):
    # Not efficient but allows keeping full history for logs...
    # lr close to 1 -> update completely to latest value
    current_estimate = hist_list[0] if hist_list else 0

    for new_observation in hist_list[1:]:
        current_estimate += lr * (new_observation - current_estimate)

    return current_estimate




def calc_from_updated_value_estimates(
        tr_innovations, full_innovations, thetas,
        history_avg_tr_innov, history_avg_full_innov, lr=1):

    n_truncated = len(tr_innovations)
    n_full = len(full_innovations) 
    if n_truncated == 0:
        print("no prev truncated samples from which to estimate innovation, using historical...")
        avg_tr_innov = np.mean(history_avg_tr_innov)
    else:
        avg_tr_innov_t1 = np.mean(tr_innovations)
        if avg_tr_innov_t1 != 0:
            history_avg_tr_innov.append(avg_tr_innov_t1)
        if lr == 1:
            avg_tr_innov = np.mean(history_avg_tr_innov)
        else:
            avg_tr_innov = update_value_estim(history_avg_tr_innov, lr=lr)
    # now for full-distrib samples:
    if n_full==0:
        avg_full_innov = np.mean(history_avg_full_innov)
        print("no prev middle samples from which to estimate innovation, using historical...")
    else:
        avg_full_innov_t1 = np.mean(full_innovations)
        if avg_full_innov_t1 != 0:
            history_avg_full_innov.append(avg_full_innov_t1)
        if lr == 1:
            avg_full_innov = np.mean(history_avg_full_innov)
        else:
            avg_full_innov = update_value_estim(history_avg_full_innov, lr=lr)
    # Calculate expected value of sample types given each individuals theta_i
    value_estimates_truncated =  vectorized_calc_benefits_fn_fullsamp(thetas, avg_tr_innov)
    value_estimates_fullsamp =  vectorized_calc_benefits_fn_truncated(thetas, avg_full_innov)

    return value_estimates_fullsamp, value_estimates_truncated, history_avg_tr_innov, history_avg_full_innov



def calc_initial_expected_sample_vals(
        nr_init_val_rounds, init_choice_vector, public_pdf, thetas, 
        t_distribution_pdf, t_degrees_freedom, sigma_tr,
        enforce_nonneg, infinite_approx_for_tails):

    history_avg_tr_innov = []
    history_avg_full_innov = []

    for init_round in range(nr_init_val_rounds):
        print("init value round %d" % init_round)
     
        tr_samples, full_samples = draw_est_samples_based_on_choices(
            init_choice_vector, t_degrees_freedom, sigma_tr, st_dev=None, 
            lower_bound=None, upper_bound=None, setseed=init_round)

        all_new_samples = np.concatenate([tr_samples, full_samples])
        # don't save these samples - just for initializing values; they aren't included in evolution of pdf.

        tr_innovations = calc_innov_for_samples(
            tr_samples, public_pdf, t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)

        full_innovations = calc_innov_for_samples(
            full_samples, public_pdf, t_distribution_pdf, infinite_approx_for_tails, enforce_nonneg)

        value_estimates_fullsamp, value_estimates_truncated, history_avg_tr_innov, history_avg_full_innov = \
            calc_from_updated_value_estimates(
                tr_innovations, full_innovations, thetas, 
                history_avg_tr_innov, history_avg_full_innov,
                lr=1
                )

        mn_tr_thisround = np.mean(value_estimates_truncated)
        print("Mn trun value init round %d: %.3f"% (init_round, mn_tr_thisround))
        history_avg_tr_innov.append(mn_tr_thisround)

        mn_full_thisround = np.mean(value_estimates_fullsamp)
        print("Mn Full value init round %d: %.3f"% (init_round, mn_full_thisround))
        history_avg_full_innov.append(mn_full_thisround)

        public_pdf = update_kde_pdf(public_pdf, all_new_samples)
    return history_avg_full_innov, history_avg_tr_innov


