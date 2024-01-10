import torch

def mean_correlation_coefficient(data1, data2):
    """
    Calculate the mean correlation coefficient between corresponding rows of two 2D tensors.
    The input tensors must have the same shape (batch, dim).

    Parameters:
    - data1 (torch.Tensor): First tensor of shape (batch, dim).
    - data2 (torch.Tensor): Second tensor of shape (batch, dim), correlated against data1.

    Returns:
    - mean_corr (float): The mean correlation coefficient across all rows.
    """
    # Ensure the input tensors have the same shape
    if data1.shape != data2.shape:
        raise ValueError("Input tensors must have the same shape.")

    batch_size = data1.shape[0]
    
    # Center the data
    data1_centered = data1 - data1.mean(dim=1, keepdim=True)
    data2_centered = data2 - data2.mean(dim=1, keepdim=True)

    # Compute the covariance between the corresponding rows
    covariance = (data1_centered * data2_centered).sum(dim=1)

    # Compute the standard deviations of the rows
    std_data1 = data1_centered.pow(2).sum(dim=1).sqrt()
    std_data2 = data2_centered.pow(2).sum(dim=1).sqrt()

    # Compute correlation coefficients for each row
    correlation_coeffs = covariance / (std_data1 * std_data2)

    # Compute the mean correlation coefficient
    mean_corr = correlation_coeffs.mean().item()

    return mean_corr

def rank_correlation_coefficient(data1, data2):
    """
    Calculate the mean Spearman's rank correlation coefficient between corresponding rows
    of two 2D tensors. The input tensors must have the same shape (batch, dim).

    Parameters:
    - data1 (torch.Tensor): First tensor of shape (batch, dim).
    - data2 (torch.Tensor): Second tensor of shape (batch, dim), correlated against data1.

    Returns:
    - mean_rank_corr (float): The mean Spearman's rank correlation coefficient across all rows.
    """
    if data1.shape != data2.shape:
        raise ValueError("Input tensors must have the same shape.")

    rank_corrs = []

    for i in range(data1.shape[0]):
        # Convert tensors to numpy arrays for scipy compatibility
        row_data1 = data1[i].detach().numpy()
        row_data2 = data2[i].detach().numpy()

        # Calculate Spearman's rank correlation for each pair of rows
        corr, _ = scipy.stats.spearmanr(row_data1, row_data2)
        rank_corrs.append(corr)

    # Compute the mean of the correlation coefficients
    mean_rank_corr = torch.tensor(rank_corrs).mean().item()

    return mean_rank_corr

if __name__ == "__main__":
    # Example usage:
    # Create random data tensors with the same shape (batch, dim)
    batch, dim = 100, 10
    data1 = torch.randn(batch, dim)
    data2 = torch.randn(batch, dim)

    # Calculate the mean correlation coefficient
    mean_corr_coef = mean_correlation_coefficient(data1, data2)
    print(mean_corr_coef)
